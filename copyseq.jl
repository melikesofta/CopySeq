"""
This example implements a sequence to sequence model with RNN encoder and
decoder, and the model learns to copy a sequence of words. The data is
organised in a sequence-per-line manner. This example can be extended into
an encoder-decoder machine translation model.
"""

module CopySeq

using Knet,AutoGrad,ArgParse,Compat
include(Pkg.dir("Knet/deprecated/src7/data/S2SData.jl"))
include(Pkg.dir("Knet/deprecated/src7/data/SequencePerLine.jl"))
function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
	("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
	("--hidden"; arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
	("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
	("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
	("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
	("--lr"; arg_type=Float64; default=1.0; help="Initial learning rate.")
	("--gclip"; arg_type=Float64; default=1.0; help="Value to clip the gradient norm at.")
	("--winit"; arg_type=Float64; default=0.01; help="Initial weights set to winit*randn().")
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
for f in o[:datafiles]
	push!(data, S2SData(f; batchsize=o[:batchsize], ftype=Float32, dense=true, dict=o[:datafiles][1]))
end
vocab = maxtoken(data[1],2)
train!(data, vocab; o=o)
end

function train!(data, vocab; o=nothing)
	params = initparams(vocab, o[:hidden], o[:embed], o[:winit], o[:atype])
	s0 = initstate(o[:atype], o[:hidden], o[:batchsize])
	#TODO: do dev and test
	for epoch=1:o[:epochs]
		@time s2s_loss = s2s_loop(params, copy(s0), data[1]; o=o)
		o[:fast] && continue
		println((:epoch,epoch,:loss,s2s_loss...))
	end
end

function s2s_loop(params, s0, data; o=nothing)
	#go over sentences one by one, at the end of each sentence
	#call s2s_eos which calculates the gradloss and updates the model
	decoding = false
	state = s0
	ystack = Any[]
	losscnt = zeros(2)
	for (x,ygold,mask) in data
		x = convert(KnetArray{Float32}, x)
		nwords = (mask == nothing ? size(x,2) : sum(mask))
		if decoding && ygold == nothing # the next sentence started
			s2s_eos(params, ystack; o=o)
			decoding = false
			empty!(ystack)
		end
		if !decoding && ygold != nothing # source ended, target sequence started
			decoding = true
			state[3]=copy(state[1])
			state[4]=copy(state[2])
		end
		if decoding && ygold != nothing # keep decoding target
			ygold = convert(KnetArray{Float32}, ygold)
			s2s_decode(params, x, ygold, mask, state, decoding, ystack, nwords; losscnt=losscnt, o=o)
		end
		if !decoding && ygold == nothing # keep encoding source
			s2s_encode(params, x, state, decoding; o=o)
		end
	end
	s2s_eos(params, ystack; o=o)
	return losscnt
end

function s2s_encode(params, x, state, decoding; o=nothing)
	encdec(params, x, state, decoding; o=nothing)
end

function s2s_decode(params, x, ygold, mask, state, decoding, ystack, nwords; losscnt=nothing, o=nothing)
	ypred = encdec(params, x, state, decoding; o=o)
	push!(ystack, (ypred, ygold, mask))
	if losscnt!=nothing
		(yrows, ycols) = size(ygold)
		losscnt[1] += loss(params, ystack; o=o)
		losscnt[2] += nwords/ycols
	end
	return ystack
end

function s2s_eos(params, ystack; o=nothing)
	println(loss(params, ystack; o=o))
	gloss = lossgradient(params, ystack; o=o)
	println(typeof(gloss))
	#println(size(gloss))
	for k in 1:length(params)
		params[k] -= o[:lr] * gloss[k]
	end
	for i = 1:length(state)
		isa(state,Value) && error("State should not be a Value.")
		state[i] = getval(state[i])
	end
end

function loss(params, ystack; o=nothing)
	total = 0.0; count = 0
	atype = typeof(getval(params[1]))
	println(size(ystack))
	for i=1:length(ystack)-1
		(ypred, ygold, mask) = ystack[i]
		if mask != 0 && ygold!=nothing && ypred!=nothing
			ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
			total += sum(ygold .* ynorm)
			count += size(ygold,1)
		end
	end
	return -total / count
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
	# here I take the transpose of x in order to get a row vector input
	# since S2SData returns the data in column vectors
	if !decoding
		emb = x' * params[5]
		h = lstm(params, state, 1, emb)
	else
		emb = x' * params[6]
		h = lstm(params, state, 3, emb)
		ypred = h[3] * params[7] .+ params[8]
		return exp(logp(ypred))
	end
end

function lstm(param,state,index,input; o=nothing)
	(hidden,cell) = (state[index],state[index+1])
	(weight,bias) = (param[index],param[index+1])
	gates = hcat(input, hidden) * weight + bias
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

end #module
