for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module CopySeq
using Knet,AutoGrad,ArgParse,Compat
include(Pkg.dir("Knet/src/distributions.jl"))
include("process.jl")

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
		("--dictfile"; help="Dictionary file, first datafile used if not specified")
		("--copy"; help="Generates a copy of the provided file")
		("--hidden"; arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
		("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
		("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
		("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
		("--lr"; arg_type=Float64; default=0.01; help="Initial learning rate.")
		("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
		("--seed"; arg_type=Int; default=42; help="Random number seed.")
		("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	end
	println(s.description)
	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true)
	println("opts=",[(k,v) for (k,v) in o]...)
	o[:seed] > 0 && srand(o[:seed])
	o[:atype] = eval(parse(o[:atype]))
	global data = Any[]
	push!(data, Data(o[:datafiles][1]; batchsize=o[:batchsize], vocabfile=nothing))
	dict = data[1].word_to_index
	vocab = length(dict)
	if length(o[:datafiles])>1
		for i=2:length(o[:datafiles])
			push!(data, Data(o[:datafiles][i]; batchsize=o[:batchsize], word_to_index=dict))
		end
	end
	model = initparams(vocab, o[:hidden], o[:embed], o[:atype])
	train(data, model, vocab, o)
	copydata = Data("cc50.en"; batchsize=1, word_to_index=dict)
	copysequence(model, initstate(o[:atype], o[:hidden], 1), vocab, copydata, o)
end

function train(data, model, vocab, o)
	state = initstate(o[:atype], o[:hidden], o[:batchsize])
	atype = o[:atype]
	losses = Array(Float32, length(data)-1)
	opts = oparams(model;lr=o[:lr], gclip=5)
	# calculate initial loss
	initial_loss = 0
	batch_cnt = 0
	println("vocab: ", vocab)
	for batch in data[1]
		converted_data = map(w->atype(w), batch)
		initial_loss += loss(model, converted_data, converted_data[2:end], copy(state), vocab, o)
		batch_cnt += 1
		empty!(converted_data)
	end
	gc()
	for i=2:length(data)
		losses[i-1] = test(data[i], model, copy(state), vocab, o)
	end
	println("epoch: 0\tloss: \t", initial_loss/batch_cnt, "\ttest_loss: \t", losses...)

	for epoch=1:o[:epochs]
		lss = 0
		batch_cnt = 0
		for batch in data[1]
			converted_data = map(w->atype(w), batch)
			gloss = lossgradient(model, converted_data, converted_data[2:end], copy(state), vocab, o)
			update!(model, gloss, opts)
			lss += loss(model, converted_data, converted_data[2:end], copy(state), vocab, o)
			batch_cnt += 1
			empty!(converted_data)
			gc()
		end
		for i=2:length(data)
			losses[i-1] = test(data[i], model, copy(state), vocab, o)
		end
		println("epoch: ", epoch, "\tloss: \t", lss/batch_cnt, "\ttest_loss: \t", losses...)
	end
end

function test(data, model, state, vocab, o)
	lss = 0
	batch_cnt = 0
	atype = o[:atype]
	for sentence in data
		converted_sentence = map(w->atype(w), sentence)
		lss += loss(model, converted_sentence, converted_sentence[2:end], state, vocab, o)
		batch_cnt += 1
		empty!(converted_sentence)
	end
	gc()
	return lss/batch_cnt
end

function initstate(atype, hidden, batchsize)
	state = Array(Any, 2)
	# don't need separate state arrays for encoder and decoder as decoder will continue to use the last state of encoder
	# ==> no need to copy state inbetween encoder and decoder now!
	state[1] = zeros(batchsize, hidden)
	state[2] = zeros(batchsize, hidden)
	return map(s->atype(s), state)
end

function initparams(vocab, hidden, embed, atype)
	init(d...) = atype(xavier(d...))
	w = Array(Any, 8)
	w[1] = init(hidden+embed, 4*hidden) # W_h_encoder
	w[2] = init(1, 4*hidden) # b_encoder

	w[3] = init(hidden+embed, 4*hidden) # W_h_decoder
	w[4] = init(1, 4*hidden) # b_decoder

	w[5] = init(vocab, embed) # W_enc_emb
	w[6] = init(vocab, embed) # W_dec_emb

	w[7] = init(hidden, vocab) # W_generate
	w[8] = init(1, vocab) # b_generate
	return map(p->convert(atype,p), w)
end

function loss(model, sentence, ysentence, state, vocab, o)
	total = 0.0; count = 0;
	for word in reverse(sentence)
		state = encdec(model, word, state)
	end
	input = sentence[end] # i.e. eos token
	for ygold in ysentence
		state, ypred = encdec(model, input, state; decoding=true)
		total += sum(ygold .* logp(ypred, 2))
		count += size(ygold, 1)
		input = ygold
	end
	return -total/count
end

lossgradient = grad(loss)

function encdec(model, word, state; decoding=false)
	if !decoding
		emb = word * model[5]
		state = lstm(model[1:2], state, emb)
		return state
	else
		emb = word * model[6]
		state = lstm(model[3:4], state, emb)
		ypred = state[1] * model[7] .+ model[8]
		return state, ypred
	end
end

function lstm(param, state, input)
	weight, bias 	= param
	hidden, cell 	= state
	h				= size(hidden, 2)
	gates			= hcat(input, hidden) * weight .+ bias
	forget			= sigm(gates[:, 1:h])
	ingate			= sigm(gates[:, 1+h:2h])
	outgate			= sigm(gates[:, 1+2h:3h])
	change			= tanh(gates[:, 1+3h:4h])
	cell			= cell .* forget + ingate .* change
	hidden			= outgate .* tanh(cell)
	return (hidden, cell)
end

oparams{T<:Number}(::KnetArray{T}; o...)=Adam(;o...)
oparams{T<:Number}(::Array{T}; o...)=Adam(;o...)
oparams(a::Associative; o...)=Dict(k=>oparams(v) for (k,v) in a)
oparams(a; o...)=map(x->oparams(x;o...), a)

function copysequence(model, ostate, vocab, copydata, o)
	atype = o[:atype]
	converted_data = Any[]
	for sentence in copydata
		state = copy(ostate)
		converted_sentence = map(w->atype(w), sentence)
		for word in reverse(converted_sentence)
			state = encdec(model, word, state)
		end
		input = atype(sentence[end]) # i.e. eos token
		for i=1:100
			state, ypred = encdec(model, input, state; decoding=true)
			ind = indmax(convert(Array{Float32}, ypred))
			ind==2 && break
			input = zeros(1, vocab)
			input[1, ind] = 1.0
			input = atype(input)
			print(copydata.index_to_word[ind], " ")
		end
		println()
	end
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module
