	?t????|@?t????|@!?t????|@	?`c"T???`c"T??!?`c"T??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?t????|@Z_&???A??W$?|@Y??%?<??*	"??~jn@2U
Iterator::Model::ParallelMapV2?D??]??!]8??7@)?D??]??1]8??7@:Preprocessing2F
Iterator::Model<?͌~4??!Zg?gy?F@)??WuV??1X?F??6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?<,Ԛ???!?=?M?D8@)"¿3??1?.? t4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate{Nz??ګ?!yLfЛ6@);5?u??1?j?5??*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice2Ƈ?˖?!s?????"@)2Ƈ?˖?1s?????"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?t??????!??@??K@)獓¼ǉ?1?a???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???-΂?!??t??@)???-΂?1??t??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??I`s??!?s???8@)%?S;?t?1?Տ? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?`c"T??I??{?\?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z_&???Z_&???!Z_&???      ??!       "      ??!       *      ??!       2	??W$?|@??W$?|@!??W$?|@:      ??!       B      ??!       J	??%?<????%?<??!??%?<??R      ??!       Z	??%?<????%?<??!??%?<??b      ??!       JCPU_ONLYY?`c"T??b q??{?\?X@