	??>Ȳ_@??>Ȳ_@!??>Ȳ_@	?"%?d???"%?d??!?"%?d??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??>Ȳ_@??pYM>@A?hq?0gW@Y?2p@KW??*	h??|??x@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??F>?x??!O??.??K@)?~T???1=ƛ??JH@:Preprocessing2U
Iterator::Model::ParallelMapV2L5????!??X߃)@)L5????1??X߃)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatyW=`2??!??W???,@)??#?????1?$`?q,'@:Preprocessing2F
Iterator::Modelެ?????!s??`?28@)"p$?`S??1S%?i??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice???֪??!???oM@)???֪??1???oM@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipКiQ??!???gL?R@)?T?^??1?{??T~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=?e?YJ??!X߃??@)=?e?YJ??1X߃??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]???T??!`c???L@)???
~{?1"?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?"%?d??I?n????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??pYM>@??pYM>@!??pYM>@      ??!       "      ??!       *      ??!       2	?hq?0gW@?hq?0gW@!?hq?0gW@:      ??!       B      ??!       J	?2p@KW???2p@KW??!?2p@KW??R      ??!       Z	?2p@KW???2p@KW??!?2p@KW??b      ??!       JCPU_ONLYY?"%?d??b q?n????X@