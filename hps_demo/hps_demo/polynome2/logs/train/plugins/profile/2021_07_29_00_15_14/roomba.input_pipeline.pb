	????Pv@????Pv@!????Pv@	???ǟ????ǟ?!???ǟ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????Pv@?x#????A?	?O?v@Y?u6????*	? ?rh]e@2F
Iterator::Model4H?Sȕ??!?M??<E@)͓k
dv??1?????9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????Pݤ?!?|-??7@)??镲??1D?/?{3@:Preprocessing2U
Iterator::Model::ParallelMapV28?-:Yj??!g?????0@)8?-:Yj??1g?????0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?g#?M)??!,@?B?w:@)?=?-??11Č7?{*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??À%??!%?N0s*@)??À%??1%?N0s*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipi????+??!??;0?L@)?f???1V6??}@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??(???~?!Du???o@)??(???~?1Du???o@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???{??!???Ҡ=@)?a?[>?r?1r?Z?8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???ǟ?I?b ??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?x#?????x#????!?x#????      ??!       "      ??!       *      ??!       2	?	?O?v@?	?O?v@!?	?O?v@:      ??!       B      ??!       J	?u6?????u6????!?u6????R      ??!       Z	?u6?????u6????!?u6????b      ??!       JCPU_ONLYY???ǟ?b q?b ??X@