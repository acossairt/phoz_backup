	BҧU??c@BҧU??c@!BҧU??c@	|Qϓl۲?|Qϓl۲?!|Qϓl۲?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$BҧU??c@??RxРN@A[??	@X@Yk??=]ݽ?*	?&1?Rp@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?????	??!?nͻ?G@)yͫ:???1?
q??uC@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?3????!???@?4@)?5??
??1?l????1@:Preprocessing2U
Iterator::Model::ParallelMapV2!??F???!? U?)@)!??F???1? U?)@:Preprocessing2F
Iterator::ModelB>?٬???!?7??Ge9@)c??l?ޠ?1N_??:;)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice????:8??!??q??"@)????:8??1??q??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipѐ?(????!????R@)??G??'??1?2??
%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??
????!F?k???@)??
????1F?k???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapD6?.6???!(o?Vk?H@)?i??&kt?1??[3???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9}Qϓl۲?I,?$I?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??RxРN@??RxРN@!??RxРN@      ??!       "      ??!       *      ??!       2	[??	@X@[??	@X@![??	@X@:      ??!       B      ??!       J	k??=]ݽ?k??=]ݽ?!k??=]ݽ?R      ??!       Z	k??=]ݽ?k??=]ݽ?!k??=]ݽ?b      ??!       JCPU_ONLYY}Qϓl۲?b q,?$I?X@