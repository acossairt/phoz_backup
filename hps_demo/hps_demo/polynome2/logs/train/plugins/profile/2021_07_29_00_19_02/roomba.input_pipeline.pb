	"臅彻\@"臅彻\@!"臅彻\@	?6?85>???6?85>??!?6?85>??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"臅彻\@??.%@Az蔫箙Z@Yr剥厦?*	濓K媨@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateN)瘯衇??!郳竇bM@)R?r??1妲鰼J@:Preprocessing2F
Iterator::Model谌uS蔾??!绶*轆?6@)1桾m7怜?1Q捛zm%,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL?^I虬?!+賀p?
.@)y苏廙颢?1~甃,M?(@:Preprocessing2U
Iterator::Model::ParallelMapV2凓?9]??!|輱A? @)凓?9]??1|輱A? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceMK瘦鐦?!支?-S@)MK瘦鐦?1支?-S@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa7l[斮??!Ru堬]S@)Na姫??1舿9T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx靏?蓢?!蔼!@)x靏?蓢?1蔼!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?询瓱??!终Eq$xM@)e峼團wp?1O=^銊0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?6?85>??Ie穋錪鱔@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??.%@??.%@!??.%@      ??!       "      ??!       *      ??!       2	z蔫箙Z@z蔫箙Z@!z蔫箙Z@:      ??!       B      ??!       J	r剥厦?r剥厦?!r剥厦?R      ??!       Z	r剥厦?r剥厦?!r剥厦?b      ??!       JCPU_ONLYY?6?85>??b qe穋錪鱔@