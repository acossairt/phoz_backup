	 9a?h?q@ 9a?h?q@! 9a?h?q@	0??(X???0??(X???!0??(X???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ 9a?h?q@W%?}?:Z@A=?N?<f@Y?]L3????*	5^?I?r@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?sa????!??3??>@)A+0du???1q?z5??9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????????!???SI6?@)
??a???1 ????6@:Preprocessing2F
Iterator::Model+???ڧ??!????6?>@)????ߩ?1???0??0@:Preprocessing2U
Iterator::Model::ParallelMapV2?7M?p??!@I{???+@)?7M?p??1@I{???+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?f??I}??!g5?m?? @)?f??I}??1g5?m?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?!? ?&??!H.??3@)?!? ?&??1H.??3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx^*6?u??!???Z?FQ@)8??????1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??lu9??!cM??Tx@@)???,u?1??8????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no90??(X???I	????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W%?}?:Z@W%?}?:Z@!W%?}?:Z@      ??!       "      ??!       *      ??!       2	=?N?<f@=?N?<f@!=?N?<f@:      ??!       B      ??!       J	?]L3?????]L3????!?]L3????R      ??!       Z	?]L3?????]L3????!?]L3????b      ??!       JCPU_ONLYY0??(X???b q	????X@