	?a?r@?a?r@!?a?r@	ݨ?D??@ݨ?D??@!ݨ?D??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?a?r@?t?_????Aڌ??-q@Y?O??`+@*	\??????@2F
Iterator::Model?s???y$@!??At?<W@)#??]$@1Bɉ?>W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9(a?m??!?d[Wt@):=?ƂB??1UY)C? @:Preprocessing2U
Iterator::Model::ParallelMapV2?{b?*߫?!6??????)?{b?*߫?16??????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???$y??!?ބ?J??)?"0?70??1?3;8>???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????9???!?K??8??)????9???1?K??8??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??f????!?s???1@)+?)?T??1?6?ɿ???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!;oc?#??!U(:?????)!;oc?#??1U(:?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ݨ?D??@Ire?۳?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?t?_?????t?_????!?t?_????      ??!       "      ??!       *      ??!       2	ڌ??-q@ڌ??-q@!ڌ??-q@:      ??!       B      ??!       J	?O??`+@?O??`+@!?O??`+@R      ??!       Z	?O??`+@?O??`+@!?O??`+@b      ??!       JCPU_ONLYYݨ?D??@b qre?۳?W@