	??m4??1@??m4??1@!??m4??1@	S?o4?#@S?o4?#@!S?o4?#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??m4??1@Ac&Q/??Ac{-??0@Y?8?:V??*	j?t|?@2F
Iterator::Model???-???!??΂?DV@)??g??s??1?V-1wcH@:Preprocessing2U
Iterator::Model::ParallelMapV2?2??(??!?oԥ%D@)?2??(??1?oԥ%D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI??Z????!'?:?e}!@)?????@??1&j????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Z???!YRt???)?7?-:Y??1f?f???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?;Mf????!???e????)?;Mf????1???e????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Sr3|?!???{ʕ??)??Sr3|?1???{ʕ??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??)?J??!?K????%@)?\????{?1bbw?A??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapӣ??????!??ێ?!@)]???Ej?1]???B
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9S?o4?#@I?r9?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ac&Q/??Ac&Q/??!Ac&Q/??      ??!       "      ??!       *      ??!       2	c{-??0@c{-??0@!c{-??0@:      ??!       B      ??!       J	?8?:V???8?:V??!?8?:V??R      ??!       Z	?8?:V???8?:V??!?8?:V??b      ??!       JCPU_ONLYYS?o4?#@b q?r9?V@