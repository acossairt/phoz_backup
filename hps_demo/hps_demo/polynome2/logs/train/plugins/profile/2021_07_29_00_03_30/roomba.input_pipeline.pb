	??Po\@??Po\@!??Po\@	???bԏ?????bԏ??!???bԏ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Po\@"?4??H@A1]??KO@Y?:?vٯ??*	43333B?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateyX?5?;??!s??T?Q@)????5w??1O???t?P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatrN??}???!?
4'#%@)˽??P???1????O?!@:Preprocessing2U
Iterator::Model::ParallelMapV2??'G???!<?6?:@)??'G???1<?6?:@:Preprocessing2F
Iterator::Model??????!????s?*@)?f׽???1Lʸ??:@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??VBwI??!0R???@)??VBwI??10R???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? ?????!?????U@)0?̕A??1?$H?
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg|_\?҆?!/2`Ӻ???)g|_\?҆?1/2`Ӻ???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapꕲq???!n?c?,R@))\???(|?1L??c???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???bԏ??I6??8?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"?4??H@"?4??H@!"?4??H@      ??!       "      ??!       *      ??!       2	1]??KO@1]??KO@!1]??KO@:      ??!       B      ??!       J	?:?vٯ???:?vٯ??!?:?vٯ??R      ??!       Z	?:?vٯ???:?vٯ??!?:?vٯ??b      ??!       JCPU_ONLYY???bԏ??b q6??8?X@