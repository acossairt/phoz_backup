	?????@?????@!?????@	Z??????Z??????!Z??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????@?tp?8??ALqUٗ??@Y?M*k??*	?????@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateS????G.@!Ԫò?W@)ܷZ'./.@1???U?W@:Preprocessing2F
Iterator::Modelq㊋#??!?y>Q?@)#0?70???1ec?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicehv?[????!A?x]??)hv?[????1A?x]??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??-u?׫?!???BG???)??F??1u??Z?!??:Preprocessing2U
Iterator::Model::ParallelMapV2???0????!TU?\T???)???0????1TU?\T???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ŏ1o.@!e???W@)kծ	i???1r?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8?:V)=??!?[bB?J??)8?:V)=??1?[bB?J??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapm??J?J.@!qƅ???W@)??P??dv?1??܀???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Z??????I?yě??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?tp?8???tp?8??!?tp?8??      ??!       "      ??!       *      ??!       2	LqUٗ??@LqUٗ??@!LqUٗ??@:      ??!       B      ??!       J	?M*k???M*k??!?M*k??R      ??!       Z	?M*k???M*k??!?M*k??b      ??!       JCPU_ONLYYZ??????b q?yě??X@