	voEb?XQ@voEb?XQ@!voEb?XQ@	*??V??*??V??!*??V??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$voEb?XQ@ZJ??P4@Aa?>?H@Y?]??y???*	gffff?|@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[^??6S??!?7]??J@)A?m??1??J??H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4???????!?x??6@)?=	l????1?04A?'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???O=??!Q??wW&@)???O=??1Q??wW&@:Preprocessing2U
Iterator::Model::ParallelMapV27?n?e??!b????#@)7?n?e??1b????#@:Preprocessing2F
Iterator::Model?k%t?ĵ?!?7>?ˈ2@)w?Nyt#??1)g???%!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??D?k???!r??]T@)x'?ے?1???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?x"??p??!s??>Tg@)?x"??p??1s??>Tg@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(?????!yߣ?i;K@)?W歺u?1R??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 28.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9*??V??I낇?v?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ZJ??P4@ZJ??P4@!ZJ??P4@      ??!       "      ??!       *      ??!       2	a?>?H@a?>?H@!a?>?H@:      ??!       B      ??!       J	?]??y????]??y???!?]??y???R      ??!       Z	?]??y????]??y???!?]??y???b      ??!       JCPU_ONLYY*??V??b q낇?v?X@