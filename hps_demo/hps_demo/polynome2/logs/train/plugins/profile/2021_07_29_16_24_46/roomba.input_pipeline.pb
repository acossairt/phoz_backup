	4???:q@4???:q@!4???:q@	e?E\@e?E\@!e?E\@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$4???:q@???&??A?b???p@YE(b?"@*	?n?`?@2U
Iterator::Model::ParallelMapV2mUه"@!E?+||T@)mUه"@1E?+||T@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?lw????!??QW?y0@);oc?#??13?Um0@:Preprocessing2F
Iterator::Model???K?"@!??4??T@)?e??S9??1}?MBn'??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??s?/??!Ω?? ???)??s?/??1Ω?? ???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?W zR&??!vL6?a??)Cr2q???1e&S?m??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??H???!s -??1@)}?|??1?A??U??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Q?????!AA??ϳ?)?Q?????1AA??ϳ?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape?/?????!?ހ?g?0@)`?U,~Sx?1???]????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9d?E\@I?G?%X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???&?????&??!???&??      ??!       "      ??!       *      ??!       2	?b???p@?b???p@!?b???p@:      ??!       B      ??!       J	E(b?"@E(b?"@!E(b?"@R      ??!       Z	E(b?"@E(b?"@!E(b?"@b      ??!       JCPU_ONLYYd?E\@b q?G?%X@