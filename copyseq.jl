"""
This example implements a sequence to sequence model with RNN encoder and
decoder, and the model learns to copy a sequence of words. The data is
organised in a sequence-per-line manner. This example can be extended into
an encoder-decoder machine translation model.
"""

module CopySeq

using Knet,AutoGrad,ArgParse,Compat
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
for f in o[:datafiles]
	push!(data, process(f, o[:batchsize], o[:atype]))
end
vocab = length(data[1][1][2])
train!(data, vocab; o=o)
end

function train!(data, vocab; o=nothing)
	params = initparams(vocab, o[:hidden], o[:embed], o[:winit], o[:atype])
	state = initstate(o[:atype], o[:hidden], o[:batchsize])
	#TODO: do dev and test
	first_loss = true
	for epoch=0:o[:epochs]
		lss = 0
		sent_cnt = 0
		for sentence in data[1]
			lss += train1(params, state, sentence, vocab; first_loss=first_loss, o=o)
			sent_cnt += 1
			if o[:gcheck] > 0 && sent_cnt == 1 #check gradients only for one sentence
				gradcheck(loss, params, sentence, copy(state), vocab; gcheck=o[:gcheck], o=o)
			end
		end
		first_loss=false
		println((:epoch,epoch,:loss,lss/sent_cnt))
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
	for i = 1:length(state)
		isa(state,Value) && error("State should not be a Value.")
		state[i] = getval(state[i])
	end
	lss = loss(params, sentence, state, vocab; o=o)
	return lss
end

function loss(params, sentence, state, vocab; o=nothing)
	#encoder
	decoding=false
	for word in sentence
		state = encdec(params, word, state, decoding; o=o)
	end

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

	for ygold in sentence
		state, ypred = encdec(params, input, state, decoding; o=o) #predict
		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		total += sum(ygold .* ynorm)
		count += size(ygold,1)
		input = ygold
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

function process(datafile, batchsize, atype)
	#TODO: optimize memory usage
	dict = readvocab(datafile)
	data, line = Any[], Any[]
	word = nothing
	open(datafile) do f
		for l in eachline(f)
			for w in split(l)
				word = zeros(Float64, batchsize, length(dict))
				word[dict[w]] = 1.0
				push!(line, copy(convert(atype, word)))
			end
			push!(data, copy(line))
			empty!(line)
		end
	end
	return data
end

function readvocab(file)
	d = Dict{Any,Int}()
	open(file) do f
		for l in eachline(f)
			for w in split(l)
				get!(d, w, 1+length(d))
			end
		end
	end
	return d
end

end #module
