import util
import numpy as np

try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	from pycuda import gpuarray
except:
	print "PyCuda not installed, or no compatible device detected"


class hf_optimizer(object):
	def __init__(self, cnf, data, info, targetchunk, logger):
		self.cnf = cnf
		self.data = data
		self.info = info
		self.targetchunk = targetchunk
		self.logger = logger


	def calc_gradient(self):
		#function [y grad grad2 gradchunk grad2chunk ll err] = calc_gradient(cnf, data, info, targetchunk)
		y = np.zeros((1,1))
		grad = np.zeros((1,1))
		grad2 = np.zeros((1,1))
		gradchunk = np.zeros((1,1))
		grad2chunk = np.zeros((1,1))
		oldll = np.zeros((1,1))
		olderr = np.zeros((1,1))
		# -------- initialize parameter

		[Wu, bu] = unpack_param(info.paramsp,cnf);

		ll = 0;
		err = 0;

		y = cell(cnf.numchunks, cnf.numlayers+1);

		grad = mzeros(cnf.psize,1);
		grad2 = mzeros(cnf.psize,1);

		gradchunk = [];
		grad2chunk = [];

		# -------- calculate gradient

		for chunk = 1:cnf.numchunks
			yi = conv(data.indata(:, ((chunk-1)*cnf.sizechunk+1):(chunk*cnf.sizechunk) ));
			outc = conv(data.outdata(:, ((chunk-1)*cnf.sizechunk+1):(chunk*cnf.sizechunk) ));

			[y_chunk yi xi] = fw_prop(cnf, Wu, bu, yi);

			ll = calc_ll(cnf, outc, yi, xi, ll);
			err = calc_err(cnf, outc, yi, xi, err);

			for i=1:numel(y_chunk)
				y{chunk, i} = y_chunk{i};
			end

			[grad grad2 gradchunk grad2chunk] = ...
				bw_prop(cnf, chunk, targetchunk, Wu, bu, outc, y, ...
				grad, grad2, gradchunk, grad2chunk);
		end

	    # -------- normalize results

		ll = ll / cnf.numcases;
		err = err / cnf.numcases;

		ll = ll - 0.5*cnf.weightcost*np.dot(np.transpose(info.paramsp),(cnf.maskp.*info.paramsp));

		grad = grad/conv(cnf.numcases) - conv(cnf.weightcost)*(cnf.maskp.*info.paramsp);
		grad2 = grad2/conv(cnf.numcases);

		gradchunk = gradchunk/conv(cnf.sizechunk) - conv(cnf.weightcost)*(cnf.maskp.*info.paramsp);
		grad2chunk = grad2chunk/conv(cnf.sizechunk);

		self.y = y
		self.grad = grad
		self.grad2 = grad2
		self.gradchunk = gradchunk
		self.grad2chunk = grad2chunk
		self.oldll = oldll
		self.olderr =olderr

	def optimize(self):
		#function [info ll err] = optimize(cnf, data, info, targetchunk, y, grad, precon, oldll)
		# -------- initialize parameter

		[Wu, bu] = unpack_param(info.paramsp,cnf);

		#slightly decay the previous change vector before using it as an
		#initialization.  This is something I didn't mention in the paper,
		#and it's not overly important but it can help a lot in some situations 
		#so you should probably use it

		info.ch = conv(cnf.decay)*info.ch;

		# -------- conjugate gradient descent

		#maxiters is the most important variable that you should try
		#tweaking.  While the ICML paper had maxiters=250 for everything
		#I've since found out that this wasn't optimal.  For example, with
		#pre-trained weights for CURVES, maxiters=150 is better.  And for
		#the FACES dataset you should use something like maxiters=100.
		#Setting it too small or large can be bad to various degrees.
		#Currently I'm trying to automate"this choice, but it's quite hard
		#to come up with a *robust* heuristic for doing this.

		maxiters = 250;
		miniters = 1;
		fprintf('maxiters: %d miniters: %d\n', maxiters, miniters);

		[chs, iterses] = conjgrad( ...
		@(V)-calc_GV(V,Wu,y,info.lmbda,cnf,targetchunk), ...
		grad, info.ch, ceil(maxiters), ceil(miniters), precon );

		info.ch = chs{end};
		iters = iterses(end);

		info.totalpasses = info.totalpasses + iters;
		fprintf('CG steps used: %d, total is: %d\n', iters, info.totalpasses);

		p = info.ch;

		fprintf('ch magnitude: %d\n', double(norm(info.ch)));

		# -------- choose ch

		j = length(chs);

		ll = [];

		for j = (length(chs)-1):-1:1
			[lowll, lowerr] = evaluate(info.paramsp + chs{j}, ...
			    data.indata, data.outdata, cnf, cnf.numchunks);

			if ll > lowll
			    j = j+1;
			    break;
			end

			ll = lowll;
			err = lowerr;
		end
		if isempty(j)
		    j = 1;
		end

		p = chs{j};

		fprintf('Chose iters: %d\n', iterses(j));

		# -------- update rho

		[ll_chunk, err_chunk] = evaluate(info.paramsp + p, ...
		data.indata, data.outdata, cnf, cnf.numchunks, targetchunk);
		[oldll_chunk, olderr_chunk] = evaluate(info.paramsp, ...
		data.indata, data.outdata, cnf, cnf.numchunks, targetchunk);

		#disabling the damping when computing rho is something I'm not 100% sure
		#about.  It probably doesn't make a huge difference either way.  Also this
		#computation could probably be done on a different subset of the training data
		#or even the whole thing

		cnf.autodamp = 0;
		denom = -0.5*double(p'*calc_GV(p,Wu,y,info.lambda,cnf,targetchunk)) - double(grad'*p);
		cnf.autodamp = 1;
		rho = (oldll_chunk - ll_chunk)/denom;
		if oldll_chunk - ll_chunk > 0
			rho = -Inf;

		fprintf('rho: %f\n', rho);

		# -------- update lambda

		#%the damping heuristic (also very standard in optimization):
		if cnf.autodamp
		    if rho < 0.25 || isnan(rho)
		        info.lmbda = info.lmbda*cnf.boost;
		    elseif rho > 0.75
		        info.lmbda = info.lmbda*cnf.drop;
		    end
		    fprintf('New lambda: %f\n', info.lmbda);
		end

		# -------- update learning rate

		#bog-standard back-tracking line-search implementation:
		rate = 1.0;

		c = 10^(-2);
		j = 0;
		while j < 60
		    if ll >= oldll + c*rate*double(np.transpose(grad)*p)
		        break;
		    else
		        rate = 0.8*rate;
		        j = j + 1;
		    end

		    #this is computed on the whole dataset.  If this is not possible you can
		    #use another set such the test set or a seperate validation set

		    [ll, err] = evaluate(info.paramsp + conv(rate)*p, ...
		        data.indata, data.outdata, cnf, cnf.numchunks);
		end

		if j == 60
		    %completely reject the step
		    j = Inf;
		    rate = 0.0;
		    ll = oldll;
		end

		fprintf('Number of reductions: %d, chosen rate: %f\n', j, rate);

		# -------- update parameter

		info.paramsp = info.paramsp + conv(rate)*p;


	def preconditioning(self):
		#function precon = preconditioning
		#preconditioning vector
		precon = (grad2 + mones(cnf.psize,1)*conv(info.lmbda) + cnf.maskp*conv(cnf.weightcost)).^(3/4);
		self.precon = precon

	def evaluate(self,indata,target,nchunks,tchunks=None):
		#function [ll, err] = evaluate(params, in, out, cnf, nchunks, tchunk)
		params = self.info.paramsp
		cnf = self.cnf
		#-------- initialize parameter
		Wu, bu = util.unpack_param(param,cnf)
		ll = 0
		err = 0
		schunk = indata.shape[1]/nchunks;

		if (tchunks != None):
			chunkrange = tchunks
		else:
			chunkrange = np.arange(0:nchunks).astype(int)

		#-------- evaluate
		for chunk in chunkrange:
			yi = indata[:, ((chunk-1)*schunk+1):(chunk*schunk) ]
			outc = target[:, ((chunk-1)*schunk+1):(chunk*schunk) ]

			y_chunk,yi,xi = self.fw_prop(Wu, bu, yi)

			ll = self.calc_ll(outc, yi, xi, ll)
			err = self.calc_err(outc, yi, xi, err)

		#-------- normalize results
		ll = ll / indata.shape[1]
		err = err / indata.shape[1]

		if not(tchunks == None):
			ll = ll*nchunks
			err = err*nchunks 

		ll = ll - 0.5*cnf.weightcost*(np.dot(np.transpose(param),(cnf.maskp*param)))
		#print ll
		return ll, err

	def bw_prop(self):
		#function [grad grad2 gradchunk grad2chunk]  = bw_prop(cnf, chunk, targetchunk, Wu, bu, outc, y, grad, grad2, gradchunk, grad2chunk)
		dEdW = cell(cnf.numlayers, 1)
		dEdb = cell(cnf.numlayers, 1)

		dEdW2 = cell(cnf.numlayers, 1)
		dEdb2 = cell(cnf.numlayers, 1)

		yi = y{chunk, cnf.numlayers+1}

		if (cnf.hybridmode) and not (chunk != targetchunk):
			y{chunk, cnf.numlayers+1} = []; #save memory

		for i = cnf.numlayers:-1:1
			if i < cnf.numlayers
				if (cnf.layertypes{i} == "logistic")
					dEdxi = dEdyi.*yi.*(1-yi);
				elif strcmp(cnf.layertypes{i}, 'tanh')
					dEdxi = dEdyi.*(1+yi).*(1-yi);
				elif strcmp(cnf.layertypes{i}, 'linear')
					dEdxi = dEdyi;
				else:
					self.logger.error('Unknown/unsupported layer type');
			else
				dEdyi = 2*(outc - yi); #mult by 2 because we dont include the 1/2 before

				if strcmp(cnf.layertypes{i}, 'logistic')
					dEdxi = dEdyi.*yi.*(1-yi);
				elif strcmp(cnf.layertypes{i}, 'tanh')
					dEdxi = dEdyi.*(1+yi).*(1-yi);
				elif strcmp(cnf.layertypes{i}, 'linear')
					dEdxi = dEdyi;
				elif strcmp(cnf.layertypes{i}, 'softmax')
					dEdxi = dEdyi.*yi - yi.* repmat(sum(dEdyi.*yi, 1), [layersizes(i+1) 1]);
				else:
					self.logger.error('Unknown/unsupported layer type')

			dEdyi = np.dot(np.transpose(Wu[i]),dEdxi);

			yi = conv(y{chunk, i});

			if (cnf.hybridmode) and (chunk != targetchunk):
				y{chunk, i} = []; #save memory

			# standard gradient comp:
			dEdW{i} = dEdxi*np.transpose(yi)
			dEdb{i} = sum(dEdxi,2)

			#gradient squared comp:
			dEdW2{i} = (dEdxi.^2)*np.transpose(yi.^2)
			dEdb2{i} = sum(dEdxi.^2,2);

		grad = grad + pack_param(dEdW, dEdb, cnf)
		grad2 = grad2 + pack_param(dEdW2, dEdb2, cnf)

		if (chunk == targetchunk):
			gradchunk = pack_param(dEdW, dEdb, cnf)
			grad2chunk = pack_param(dEdW2, dEdb2, cnf)
		
		self.grad = grad
		self.grad2 = grad2
		self.gradchunk = gradchunk
		self.grad2chunk = self.grad2chunk

	def calc_err(self): 
		#function err = calc_err(cnf, outc, yi, xi, err)
		cnf = self.cnf
		outc = self.outc
		yi = self.yi
		xi = self.xi
		err = self.err

		if (cnf.errtype == 'class'):
			err = err + sum( sum(outc.*yi,1) ~= max(yi,[],1) ) 
		elif (cnf.errtype == 'L2'):
			err = err + sum(sum((yi - outc).^2, 1))
		elif (cnf.errtype ==  'none'):
			a = 0  	#do nothing
		else:
			self.info.error("Unrecognized error type")
		self.err = err

	def calc_GV(self):
		#function GV = calc_GV(V, Wu, y, lambda, cnf, targetchunk)
		[VWu, Vbu] = unpack_param(V,cnf);

		GV = mzeros(cnf.psize,1);

		if (cnf.hybridmode):
			chunkrange = targetchunk
		else:
			chunkrange = 1:numchunks

		#application of R operator
		for chunk = chunkrange
			Ry_chunk = rfw_prop(cnf, chunk, Wu, VWu, Vbu, y);

			for i=1:numel(Ry_chunk)
				Ry{chunk, i} = Ry_chunk{i};

			[GVW, GVb] = rbw_prop(cnf, chunk, Wu, VWu, Vbu, y, Ry);

			GV = GV + pack_param(GVW, GVb, cnf);

		GV = GV / conv(cnf.numcases);

		if cnf.hybridmode
			GV = GV * conv(cnf.numchunks);


		GV = GV - conv(cnf.weightcost)*(cnf.maskp.*V);

		if (cnf.autodamp):
			GV = GV - conv(lambda)*V;

	def calc_ll(self):
		#function ll = calc_ll(cnf, outc, yi, xi, ll)
		if strcmp( cnf.layertypes{cnf.numlayers}, 'linear' )
			ll = ll + double( -sum(sum((outc - yi).^2)) );
		elseif strcmp( cnf.layertypes{cnf.numlayers}, 'logistic' )
			ll = ll + sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));
		elseif strcmp( cnf.layertypes{cnf.numlayers}, 'softmax' )
			ll = ll + double(sum(sum(outc.*log(yi))));
		end


	def conjgrad(self):
		#function [xs, is] = conjgrad( Afunc, b, x0, maxiters, miniters, Mdiag )
		tolerance = 5e-4;

		gapratio = 0.1;
		mingap = 10;

		maxtestgap = max(ceil(maxiters * gapratio), mingap) + 1;

		vals = zeros(maxtestgap,1);

		inext = 5;
		imult = 1.3;

		is = [];
		xs = {};

		r = Afunc(x0) - b;
		y = r./Mdiag;

		p = -y;
		x = x0;

		#val is the value of the quadratic model
		val = 0.5*double(np.transpose(-b+r)*x);
		#disp( ['iter ' num2str(0) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', val = ' num2str( val ) ]);

		for i = 1:maxiters

        #compute the matrix-vector product.  This is where 95% of the work in
        #HF lies:
			Ap = Afunc(p);

			pAp = np.transpose(p)*Ap;

			#the Gauss-Newton matrix should never have negative curvature.  The
			#Hessian easily could unless your objective is convex
			if pAp <= 0
				self.logger.warning('Negative Curvature!')
				self.logger.warning('Bailing...')
				break

			alpha = (np.transpose(r)*y)/pAp;

			x = x + alpha*p;
			r_new = r + alpha*Ap;

			y_new = r_new./Mdiag;

			beta = (np.transpose(r_new)*y_new)/(np.transpose(r)*y);

			p = -y_new + beta*p;

			r = r_new;
			y = y_new;

			val = 0.5*double(np.transpose(-b+r)*x);
			vals( mod(i-1, maxtestgap)+1 ) = val;

			#disp( ['iter ' num2str(i) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', val = ' num2str( val ) ]);

			testgap = max(ceil( i * gapratio ), mingap);
			prevval = vals( mod(i-testgap-1, maxtestgap)+1 ); %testgap steps ago

			if i == ceil(inext)
			    is(end+1) = i;
			    xs{end+1} = x;
			    inext = inext*imult;
			end

			#the stopping criterion here becomes largely unimportant once you
			#optimize your function past a certain point, as it will almost never
			#kick in before you reach i = maxiters.  And if the value of maxiters
			#is set so high that this never occurs, you probably have set it too
			#high
			if i > testgap && prevval < 0 && (val - prevval)/val < tolerance*testgap && i >= miniters
			    break;
			end


		if i ~= ceil(inext)
			is(end+1) = i;
			xs{end+1} = x;
		end

	def fw_prop(self):
		#function [y yi xi] = fw_prop(cnf, Wu, bu, yi)
		y{1} = conv(yi);

		for i = 1:cnf.numlayers
			xi = Wu{i}*yi + repmat(bu{i}, [1 size(yi,2)]);

			if strcmp(cnf.layertypes{i}, 'logistic')
				yi = 1./(1 + exp(-xi));
			elseif strcmp(cnf.layertypes{i}, 'tanh')
				yi = tanh(xi);
			elseif strcmp(cnf.layertypes{i}, 'linear')
				yi = xi;
			elseif strcmp( cnf.layertypes{i}, 'softmax' )
				tmp = exp(xi);
				yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );
				tmp = [];
			else
				error( 'Unknown/unsupported layer type' );
			end

			y{i+1} = conv(yi);
		end

	def rbw_pro(self):
		#function [GVW GVb] = rbw_prop(cnf, chunk, Wu, VWu, Vbu, y, Ry)
		GVW = cell(cnf.numlayers,1);
		GVb = cell(cnf.numlayers,1);

		yi = conv(y{chunk, cnf.numlayers+1});
		Ryi = conv(Ry{chunk, cnf.numlayers+1});

		# R-Backward prop:
		for i = cnf.numlayers:-1:1
		    if i < cnf.numlayers
		        if strcmp(cnf.layertypes{i}, 'logistic')
		            RdEdxi = RdEdyi.*yi.*(1-yi);
		        elseif strcmp(cnf.layertypes{i}, 'tanh')
		            RdEdxi = RdEdyi.*(1+yi).*(1-yi);
		        elseif strcmp(cnf.layertypes{i}, 'linear')
		            RdEdxi = RdEdyi;
		        else
		            error( 'Unknown/unsupported layer type' );
		        end
		    else
		        RdEdyi = -2*Ryi;

		        if strcmp(cnf.layertypes{i}, 'logistic')
		            RdEdxi = RdEdyi.*yi.*(1-yi);
		        elseif strcmp(cnf.layertypes{i}, 'tanh')
		            RdEdxi = RdEdyi.*(1+yi).*(1-yi);
		        elseif strcmp(cnf.layertypes{i}, 'linear')
		            RdEdxi = RdEdyi;
		        elseif strcmp(cnf.layertypes{i}, 'softmax')
		            error( 'RMS error not supported with softmax output' );
		        else
		            error( 'Unknown/unsupported layer type' );
		        end
		    end

		    RdEdyi = np.transpose(Wu{i})*RdEdxi;

		    yi = conv(y{chunk, i});

		    GVW{i} = RdEdxi*np.transpose(yi);
		    GVb{i} = sum(RdEdxi,2);
		end

	def rbw_pro(self):
		#function Ry = rfw_prop(cnf, chunk, Wu, VWu, Vbu, y)
		Ry = cell(cnf.numlayers,1);

		yi = conv(y{chunk, 1});
		Ryi = mzeros(cnf.layersizes(1), cnf.sizechunk);

		Ry{1} = conv(Ryi);

		# R-Forward prop:
		for i = 1:cnf.numlayers
		    Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 cnf.sizechunk]);

		    yi = conv(y{chunk, i+1});

		    if strcmp(cnf.layertypes{i}, 'logistic')
		        Ryi = Rxi.*yi.*(1-yi);
		    elseif strcmp(cnf.layertypes{i}, 'tanh')
		        Ryi = Rxi.*(1+yi).*(1-yi);
		    elseif strcmp(cnf.layertypes{i}, 'linear')
		        Ryi = Rxi;
		    elseif strcmp( cnf.layertypes{i}, 'softmax' )
		        Ryi = Rxi.*yi ...
		            - yi .* repmat(sum(Rxi.*yi, 1), [cnf.layersizes(i+1) 1]);
		    else
		        error( 'Unknown/unsupported layer type' );
		    end
		    Ry{i+1} = conv(Ryi);
		end	