"""
This example implements a sequence to sequence model with RNN encoder and
decoder, and the model learns to copy a sequence of words. The data is
organised in a sequence-per-line manner. This example can be extended into
an encoder-decoder machine translation model.
"""
module CopySeq
using Knet,AutoGrad,ArgParse,Compat
include("S2SData.jl")
include("SequencePerLine.jl")
function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
	("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
  ("--dictfile"; help="Dictionary file, first datafile used if not specified")
	("--hidden"; arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
	("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
	("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
	("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
	("--lr"; arg_type=Float64; default=1.0; help="Initial learning rate.")
	("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
	("--winit"; arg_type=Float64; default=0.01; help="Initial weights set to winit*randn().")
	("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
	("--seed"; arg_type=Int; default=42; help="Random number seed.")
	("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	("--fast"; action=:store_true; help="skip loss printing for faster run")
end
println(s.description)
isa(args, AbstractString) && (args=split(args))
o = parse_args(args, s; as_symbols=true)
println("opts=",[(k,v) for (k,v) in o]...)
o[:seed] > 0 && srand(o[:seed])
o[:atype] = eval(parse(o[:atype]))
global data = Any[]
dict = (o[:dictfile] == nothing ? o[:datafiles][1] : o[:dictfile])
for f in o[:datafiles]
	push!(data, S2SData(f; batchsize=o[:batchsize], atype=o[:atype], dense=true, dict=dict))
end
vocab = maxtoken(data[1], 2)
train!(data, vocab; o=o)
end

function train!(data, vocab; o=nothing)
	params = initparams(vocab, o[:hidden], o[:embed], o[:winit], o[:atype])
	state = initstate(o[:atype], o[:hidden], o[:batchsize])
	#TODO: do dev and test
	first_loss = true
	for epoch=0:o[:epochs]
		lss = 0
		batch_cnt = 0
		sentence = Any[]
		decoding=false
		for (x, ygold, mask) in data[1]
	    if decoding && ygold == nothing # the next sentence started
				if length(sentence)!=0
					lss += train1(params, state, sentence, vocab; first_loss=first_loss, o=o)
					batch_cnt += 1
					if o[:gcheck] > 0 && batch_cnt == 1 #check gradients only for one batch
						gradcheck(loss, params, sentence, copy(state), vocab; gcheck=o[:gcheck], o=o)
					end
		      empty!(sentence)
				end
	    	decoding = false
	    end
      if !decoding && ygold != nothing # source ended, target sequence started
					decoding=true
			end
			if (!decoding && ygold == nothing) || (decoding && ygold != nothing)
				push!(sentence, (x,ygold,mask))
      end
  	end
		first_loss=false

		losses=Array(Float32, length(data)-1)
		for i=2:length(data)
			losses[i-1] = test(data[i], params, state, vocab; o=o)
		end

		println((:epoch,epoch,:trn_loss,lss/batch_cnt, :test_loss, losses...))
	end
end

function train1(params, state, sentence, vocab; first_loss=false, o=nothing)
	gloss = lossgradient(params, sentence, state, vocab; o=o)
	gscale = o[:lr]
	gclip = o[:gclip]
	if gclip > 0
		gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
		if gnorm > gclip
			gscale *= gclip / gnorm
		end
	end
	for k in 1:length(params)
		first_loss && break
		params[k] -= gscale * gloss[k]
	end
#	for i = 1:length(state)
#		isa(state,Value) && error("State should not be a Value.")
#		state[i] = getval(state[i])
#	end
	lss = loss(params, sentence, state, vocab; o=o)
	return lss
end

function test(data, params, state, vocab; o=nothing)
	lss = 0
	batch_cnt = 0
	sentence = Any[]
	decoding=false
	for (x, ygold, mask) in data
		if decoding && ygold == nothing # the next sentence started
			if length(sentence)!=0
				lss += loss(params, sentence, state, vocab; o=o)
				batch_cnt += 1
				empty!(sentence)
			end
			decoding = false
		end
		if !decoding && ygold != nothing # source ended, target sequence started
				decoding=true
		end
		if (!decoding && ygold == nothing) || (decoding && ygold != nothing)
			push!(sentence, (x,ygold,mask))
		end
	end
	return lss/batch_cnt
end

function loss(params, sentence, state, vocab; o=nothing)
	#encoder
	decoding=false
	total = 0.0
	count = 0
	input = nothing
	for (x, ygold, mask) in sentence
		#println(mask)
		if !decoding && ygold == nothing # keep encoding source
				state = encdec(params, x, state, decoding; o=o)
		elseif !decoding && ygold != nothing # source ended, target sequence started
				# copy encoder's hidden states to decoder
				state[3]=copy(state[1])
				state[4]=copy(state[2])
				decoding=true
				# give <eos> as the first token into decoder
				ypred = zeros(o[:batchsize], vocab)
				ypred[vocab] = 1
				ypred = convert(o[:atype], ypred)

				total = 0.0; count = 0 # for loss calculations
				input = ypred
		elseif decoding && ygold != nothing # keep decoding target
				state, ypred = encdec(params, input, state, decoding; o=o) #predict
				ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
				total += sum(ygold .* ynorm)
				count += size(ygold,1)
				input = ygold
		end
  end
	return -total/count
end

lossgradient = grad(loss)

function initparams(vocab, hidden, embedding, winit, atype)
	w = Any[]

	push!(w, winit*randn(hidden+embedding, hidden*4))#W_h_encoder
	push!(w, zeros(1, hidden*4))#b_encoder

	push!(w, winit*randn(hidden+embedding, hidden*4))#W_h_decoder
	push!(w, zeros(1, hidden*4))#b_decoder

	push!(w, winit*randn(vocab, embedding)) # W_enc_emb
	push!(w, winit*randn(vocab, embedding)) #	W_dec_emb

	push!(w, winit*randn(hidden, vocab))#W_generate
	push!(w, zeros(1, vocab))#b_generate
	return map(p->convert(atype,p), w)
end

function initstate(atype, hidden, batchsize)
	state = Array(Any, 4)
	for k = 1:2 # 1 for encoder 2 for decoder
		state[2k-1] = zeros(batchsize, hidden)
		state[2k] = zeros(batchsize, hidden)
	end
	return map(s->convert(atype,s), state)
end

function encdec(params, x, state, decoding; o=nothing)
	if !decoding
		emb = x * params[5]
		state = lstm(params, state, 1, emb)
	else
		emb = x * params[6]
		state = lstm(params, state, 3, emb)
		ypred = state[3] * params[7] .+ params[8]
		return state, ypred
	end
end

function lstm(param,state,index,input; o=nothing)
	(hidden,cell) = (state[index],state[index+1])
	(weight,bias) = (param[index],param[index+1])
	gates = hcat(input, hidden) * weight .+ bias
	hsize = size(hidden, 2)
	forget  = sigm(gates[:,1:hsize])
	ingate  = sigm(gates[:,1+hsize:2hsize])
	outgate = sigm(gates[:,1+2hsize:3hsize])
	change  = tanh(gates[:,1+3hsize:end])
	cell    = cell .* forget + ingate .* change
	hidden  = outgate .* tanh(cell)
	(state[index],state[index+1]) = (hidden,cell)
	return state
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module

#TODO:
#	optimize test, train and loss s2s_loops
# solve S2SData memory problem
# include JLD functionalities
