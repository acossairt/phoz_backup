	pA??5?@pA??5?@!pA??5?@	|hBk???|hBk???!|hBk???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$pA??5?@6;R}????A?V?.?@Y	????@*	R??k??@2F
Iterator::Model???>E@!?3S?M@)?_?+??@1?'???@G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatebi?G5L@!?????C@)N_??,@1 ??|??C@:Preprocessing2U
Iterator::Model::ParallelMapV2a⏢?\??!މ??X)@)a⏢?\??1މ??X)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,?PO???!,"??????)L?'????1?k?b??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?	?8???!U1"?Z??)?	?8???1U1"?Z??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipw???s?@!?5̬?hD@)8M?p]??1?P??nY??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??@?ȃ?!????P???)??@?ȃ?1????P???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\9{g?U@!v?_??C@)???5??r?1cLr?㹫?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9|hBk???I?པ(?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6;R}????6;R}????!6;R}????      ??!       "      ??!       *      ??!       2	?V?.?@?V?.?@!?V?.?@:      ??!       B      ??!       J		????@	????@!	????@R      ??!       Z		????@	????@!	????@b      ??!       JCPU_ONLYY|hBk???b q?པ(?X@