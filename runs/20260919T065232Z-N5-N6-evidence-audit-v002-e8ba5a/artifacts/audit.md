# N5–N9 evidence audit

Audit: PASS

## 20260919T055706Z-N5-screen-v5e-v004-25842c

Archive integrity: PASS

Status counts: `{"ok": 1234, "oom": 78}`

- c_output_accumulator_128_512_512 vs native_xla: {"counts": {"inconclusive": 1, "loss": 2}, "median_ratio": 0.6695613584587626, "min_ratio": 0.4966710825414175, "max_ratio": 0.917444191443233}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_128_512_512: {"counts": {"inconclusive": 1, "win": 3}, "median_ratio": 2.4545622463287216, "min_ratio": 1.120515168162119, "max_ratio": 2.989993932173078}
- s_output_accumulator_1024_1024_1024 vs native_xla: {"counts": {"inconclusive": 1}, "median_ratio": 1.0280101324543736, "min_ratio": 1.0280101324543736, "max_ratio": 1.0280101324543736}
- native_xla vs c_output_accumulator_128_512_512: {"counts": {"inconclusive": 1, "win": 2}, "median_ratio": 1.4935151011430248, "min_ratio": 1.0899845563651107, "max_ratio": 2.0134049175625397}
- s_plain_1024_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0198609649093022, "min_ratio": 1.001977871608208, "max_ratio": 1.0377440582103963}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.0818339908491803, "min_ratio": 1.0818339908491803, "max_ratio": 1.0818339908491803}
- s_interleaved_output_accumulator_2048_2048_512 vs c_plain_512_1024_1024: {"counts": {"loss": 1, "win": 1}, "median_ratio": 1.0267491698875644, "min_ratio": 0.9666392724085033, "max_ratio": 1.0868590673666254}
- s_plain_2048_2048_512 vs c_plain_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 0.9982880705246695, "min_ratio": 0.9982880705246695, "max_ratio": 0.9982880705246695}
- s_plain_512_1024_512 vs c_output_accumulator_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0158410392979629, "min_ratio": 1.0158410392979629, "max_ratio": 1.0158410392979629}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_1024_2048_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0517329412649359, "min_ratio": 1.0517329412649359, "max_ratio": 1.0517329412649359}
- s_output_accumulator_2048_1024_512 vs c_plain_512_512_256: {"counts": {"inconclusive": 1}, "median_ratio": 1.018140626717512, "min_ratio": 1.018140626717512, "max_ratio": 1.018140626717512}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_plain_1024_512_512: {"counts": {"inconclusive": 1, "loss": 1, "win": 1}, "median_ratio": 0.9418899490931417, "min_ratio": 0.8927620140312775, "max_ratio": 1.14201137718537}
- s_output_accumulator_1024_1024_512 vs c_plain_512_1024_512: {"counts": {"inconclusive": 1, "loss": 1}, "median_ratio": 0.7889738812807012, "min_ratio": 0.5458591255009828, "max_ratio": 1.0320886370604196}
- s_plain_1024_1024_1024 vs c_output_accumulator_1024_2048_512: {"counts": {"inconclusive": 1}, "median_ratio": 0.9501940060585249, "min_ratio": 0.9501940060585249, "max_ratio": 0.9501940060585249}
- s_plain_2048_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"inconclusive": 1, "win": 1, "loss": 1}, "median_ratio": 0.9730079907237021, "min_ratio": 0.5921057579936029, "max_ratio": 1.0630015565334106}
- s_interleaved_2048_1024_512 vs c_plain_2048_1024_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0320652086388626, "min_ratio": 0.9918002142149829, "max_ratio": 1.0723302030627424}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_512_512_256: {"counts": {"loss": 1}, "median_ratio": 0.9045693426877573, "min_ratio": 0.9045693426877573, "max_ratio": 0.9045693426877573}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_1024_512_512: {"counts": {"inconclusive": 2, "win": 1}, "median_ratio": 1.0171023289347934, "min_ratio": 0.9236532438705494, "max_ratio": 1.111906169253798}
- s_interleaved_output_accumulator_1024_1024_512 vs c_plain_128_512_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.6591763321325415, "min_ratio": 1.0074113455332325, "max_ratio": 2.3109413187318504}
- s_interleaved_2048_2048_512 vs c_output_accumulator_2048_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.9312604089563286, "min_ratio": 0.9312604089563286, "max_ratio": 0.9312604089563286}
- s_output_accumulator_512_1024_512 vs c_plain_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.1096810696087713, "min_ratio": 1.1096810696087713, "max_ratio": 1.1096810696087713}
- s_interleaved_512_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"inconclusive": 1, "loss": 1, "win": 1}, "median_ratio": 1.0164500800002956, "min_ratio": 0.9504532598709219, "max_ratio": 1.7834174276696932}
- s_interleaved_1024_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 3}, "median_ratio": 1.1577803161304203, "min_ratio": 1.1314011674205051, "max_ratio": 1.2602276277487687}
- c_plain_2048_2048_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.8156935970422006, "min_ratio": 0.8156935970422006, "max_ratio": 0.8156935970422006}
- s_interleaved_512_1024_512 vs c_plain_2048_2048_512: {"counts": {"win": 2}, "median_ratio": 1.2549866832046834, "min_ratio": 1.2184499593165177, "max_ratio": 1.2915234070928492}
- s_interleaved_512_1024_512 vs native_xla: {"counts": {"inconclusive": 1}, "median_ratio": 0.9938818301308132, "min_ratio": 0.9938818301308132, "max_ratio": 0.9938818301308132}
- native_xla vs c_plain_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.2259505329281923, "min_ratio": 1.2259505329281923, "max_ratio": 1.2259505329281923}
- s_plain_1024_1024_512 vs c_plain_512_1024_1024: {"counts": {"inconclusive": 1, "loss": 1, "win": 1}, "median_ratio": 1.0216534571558191, "min_ratio": 0.5499897677156391, "max_ratio": 1.051818295343862}
- s_plain_2048_2048_512 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.8701808511147141, "min_ratio": 0.8701808511147141, "max_ratio": 0.8701808511147141}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"win": 2}, "median_ratio": 1.0315987321298512, "min_ratio": 1.0093645872212544, "max_ratio": 1.0538328770384477}
- s_output_accumulator_1024_1024_1024 vs c_plain_1024_1024_1024: {"counts": {"inconclusive": 1}, "median_ratio": 0.9675541303078756, "min_ratio": 0.9675541303078756, "max_ratio": 0.9675541303078756}
- s_interleaved_output_accumulator_512_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"inconclusive": 2}, "median_ratio": 1.0124569301357509, "min_ratio": 0.9707379751859689, "max_ratio": 1.0541758850855327}
- s_interleaved_2048_1024_512 vs c_plain_128_512_512: {"counts": {"loss": 2}, "median_ratio": 0.7910231257400336, "min_ratio": 0.6863658172853698, "max_ratio": 0.8956804341946975}
- s_plain_512_1024_512 vs c_plain_1024_1024_512: {"counts": {"inconclusive": 1, "loss": 1, "win": 1}, "median_ratio": 0.9712751328245429, "min_ratio": 0.9550111591980903, "max_ratio": 1.824063824800676}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"inconclusive": 2, "win": 1}, "median_ratio": 0.9987142018592583, "min_ratio": 0.9555980057024125, "max_ratio": 1.0234531112633025}
- s_interleaved_2048_2048_512 vs c_plain_512_512_256: {"counts": {"inconclusive": 1}, "median_ratio": 0.9103796119359107, "min_ratio": 0.9103796119359107, "max_ratio": 0.9103796119359107}
- s_plain_1024_1024_1024 vs c_plain_2048_1024_512: {"counts": {"win": 3}, "median_ratio": 1.1761868541639613, "min_ratio": 1.167504942750994, "max_ratio": 1.924692691010141}
- s_plain_2048_1024_512 vs c_output_accumulator_1024_1024_512: {"counts": {"loss": 3, "inconclusive": 2}, "median_ratio": 0.9278530380671713, "min_ratio": 0.5981761772828977, "max_ratio": 1.0151760484652324}
- s_output_accumulator_2048_1024_512 vs c_plain_1024_2048_512: {"counts": {"win": 2}, "median_ratio": 1.0471430040549068, "min_ratio": 1.0453346301516042, "max_ratio": 1.0489513779582096}
- s_interleaved_output_accumulator_2048_2048_512 vs c_output_accumulator_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.8918104803998498, "min_ratio": 0.8918104803998498, "max_ratio": 0.8918104803998498}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_128_512_512: {"counts": {"loss": 3}, "median_ratio": 0.7357202922076128, "min_ratio": 0.7280640608700988, "max_ratio": 0.8824073492039258}
- s_interleaved_output_accumulator_2048_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.013819101484543, "min_ratio": 1.013819101484543, "max_ratio": 1.013819101484543}
- s_output_accumulator_512_1024_512 vs c_plain_512_1024_512: {"counts": {"inconclusive": 2, "win": 1}, "median_ratio": 1.0222000701857892, "min_ratio": 1.0020611725182131, "max_ratio": 1.042477815301219}
- c_output_accumulator_2048_1024_512 vs native_xla: {"counts": {"loss": 3, "inconclusive": 1}, "median_ratio": 0.8552115824384301, "min_ratio": 0.6887219604489966, "max_ratio": 0.9982356057267041}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0831268902058862, "min_ratio": 1.0831268902058862, "max_ratio": 1.0831268902058862}
- s_output_accumulator_2048_1024_512 vs native_xla: {"counts": {"loss": 1, "inconclusive": 1}, "median_ratio": 0.8776525682229517, "min_ratio": 0.7459732752376229, "max_ratio": 1.0093318612082804}
- native_xla vs c_output_accumulator_2048_1024_512: {"counts": {"win": 3, "inconclusive": 1}, "median_ratio": 1.1867232103774379, "min_ratio": 1.0017675128628691, "max_ratio": 1.451964736753962}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_1024_1024_1024: {"counts": {"inconclusive": 1}, "median_ratio": 0.9544148457368823, "min_ratio": 0.9544148457368823, "max_ratio": 0.9544148457368823}
- s_plain_2048_2048_512 vs c_plain_2048_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0025648552295179, "min_ratio": 1.0025648552295179, "max_ratio": 1.0025648552295179}
- s_interleaved_2048_2048_512 vs c_plain_1024_512_512: {"counts": {"loss": 2}, "median_ratio": 0.8647981351087142, "min_ratio": 0.850913443910839, "max_ratio": 0.8786828263065893}
- s_output_accumulator_512_1024_512 vs c_plain_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.1340404461057167, "min_ratio": 1.1340404461057167, "max_ratio": 1.1340404461057167}
- s_plain_512_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0746750702061303, "min_ratio": 1.0746750702061303, "max_ratio": 1.0746750702061303}
- s_interleaved_output_accumulator_2048_2048_512 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 2, "win": 1}, "median_ratio": 0.8223169412001636, "min_ratio": 0.5386825213342445, "max_ratio": 1.1873143457389408}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"inconclusive": 1}, "median_ratio": 0.9582380330994767, "min_ratio": 0.9582380330994767, "max_ratio": 0.9582380330994767}
- s_plain_2048_1024_512 vs c_plain_1024_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.847464316902416, "min_ratio": 0.847464316902416, "max_ratio": 0.847464316902416}
- s_plain_1024_1024_1024 vs c_plain_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.9062255759380526, "min_ratio": 0.9062255759380526, "max_ratio": 0.9062255759380526}
- s_interleaved_1024_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"loss": 1}, "median_ratio": 0.923484646313692, "min_ratio": 0.923484646313692, "max_ratio": 0.923484646313692}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_plain_512_512_256: {"counts": {"loss": 2}, "median_ratio": 0.7204241399903237, "min_ratio": 0.5108035727409589, "max_ratio": 0.9300447072396886}
- s_interleaved_1024_1024_1024 vs c_plain_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.2512181226733465, "min_ratio": 1.2512181226733465, "max_ratio": 1.2512181226733465}
- s_interleaved_512_1024_512 vs c_plain_512_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 0.9505630998662676, "min_ratio": 0.9505630998662676, "max_ratio": 0.9505630998662676}
- s_plain_1024_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 1, "loss": 1}, "median_ratio": 1.0756418464867832, "min_ratio": 0.9688881046686518, "max_ratio": 1.1823955883049144}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.016273589074896, "min_ratio": 1.016273589074896, "max_ratio": 1.016273589074896}
- s_interleaved_output_accumulator_2048_2048_512 vs c_output_accumulator_2048_1024_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.023415493069909, "min_ratio": 1.0145508730130646, "max_ratio": 1.0322801131267536}
- s_interleaved_output_accumulator_2048_2048_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.7625270789348493, "min_ratio": 0.7625270789348493, "max_ratio": 0.7625270789348493}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_output_accumulator_1024_1024_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0409856959834927, "min_ratio": 1.017428983590022, "max_ratio": 1.0645424083769632}
- s_plain_2048_2048_512 vs c_plain_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.8034962140578968, "min_ratio": 0.8034962140578968, "max_ratio": 0.8034962140578968}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.8966602552071352, "min_ratio": 0.8966602552071352, "max_ratio": 0.8966602552071352}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"loss": 3}, "median_ratio": 0.7599078496675645, "min_ratio": 0.5996306295813293, "max_ratio": 0.8619919173078417}
- s_plain_2048_1024_512 vs c_plain_128_512_512: {"counts": {"loss": 1}, "median_ratio": 0.7674654073264414, "min_ratio": 0.7674654073264414, "max_ratio": 0.7674654073264414}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"inconclusive": 1}, "median_ratio": 0.917174720251637, "min_ratio": 0.917174720251637, "max_ratio": 0.917174720251637}
- s_output_accumulator_512_1024_512 vs c_plain_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.1324757528529685, "min_ratio": 1.1324757528529685, "max_ratio": 1.1324757528529685}
- s_interleaved_output_accumulator_2048_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.885578984364199, "min_ratio": 0.885578984364199, "max_ratio": 0.885578984364199}
- s_plain_512_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"win": 1, "inconclusive": 1}, "median_ratio": 1.0785280296316544, "min_ratio": 0.999688860555134, "max_ratio": 1.1573671987081748}
- s_interleaved_output_accumulator_1024_1024_512 vs c_plain_1024_1024_1024: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0196887427876582, "min_ratio": 1.0152930540907037, "max_ratio": 1.0240844314846127}
- s_output_accumulator_1024_1024_1024 vs c_plain_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.9360988503902252, "min_ratio": 0.9360988503902252, "max_ratio": 0.9360988503902252}
- s_interleaved_output_accumulator_512_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.0443640010656132, "min_ratio": 1.0443640010656132, "max_ratio": 1.0443640010656132}
- s_interleaved_2048_1024_512 vs c_plain_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.8182747006943963, "min_ratio": 0.8182747006943963, "max_ratio": 0.8182747006943963}
- s_plain_1024_1024_512 vs c_plain_512_512_256: {"counts": {"loss": 1}, "median_ratio": 0.9614339293834117, "min_ratio": 0.9614339293834117, "max_ratio": 0.9614339293834117}
- c_plain_512_1024_512 vs native_xla: {"counts": {"win": 1, "inconclusive": 1}, "median_ratio": 1.039697539643484, "min_ratio": 1.006493852769248, "max_ratio": 1.0729012265177202}
- s_interleaved_1024_1024_1024 vs c_plain_512_1024_512: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.9419121849494743, "min_ratio": 0.7878287072856728, "max_ratio": 1.0959956626132756}
- s_interleaved_1024_1024_1024 vs native_xla: {"counts": {"loss": 1, "inconclusive": 1}, "median_ratio": 0.9098716907000759, "min_ratio": 0.8452623863326683, "max_ratio": 0.9744809950674836}
- native_xla vs c_plain_512_1024_512: {"counts": {"loss": 1, "inconclusive": 1}, "median_ratio": 0.9628001466104701, "min_ratio": 0.932052247946129, "max_ratio": 0.9935480452748113}
- s_interleaved_1024_1024_512 vs c_plain_512_1024_1024: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.9066903278774778, "min_ratio": 0.7540514136818429, "max_ratio": 1.0593292420731129}
- s_plain_1024_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.7680748668108576, "min_ratio": 0.7680748668108576, "max_ratio": 0.7680748668108576}
- s_interleaved_512_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.1017666561101693, "min_ratio": 1.1017666561101693, "max_ratio": 1.1017666561101693}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_output_accumulator_128_512_512: {"counts": {"win": 1}, "median_ratio": 1.1189967414056592, "min_ratio": 1.1189967414056592, "max_ratio": 1.1189967414056592}
- s_plain_2048_2048_512 vs c_plain_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.7161030919711602, "min_ratio": 0.7161030919711602, "max_ratio": 0.7161030919711602}
- s_interleaved_output_accumulator_1024_1024_512 vs c_plain_1024_512_512: {"counts": {"win": 2}, "median_ratio": 1.062459957502162, "min_ratio": 1.0525820584188739, "max_ratio": 1.0723378565854504}
- s_plain_2048_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"win": 2}, "median_ratio": 1.0248085060938545, "min_ratio": 1.0226136347128807, "max_ratio": 1.0270033774748286}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.6771031583372746, "min_ratio": 0.6771031583372746, "max_ratio": 0.6771031583372746}
- s_interleaved_2048_2048_512 vs c_plain_1024_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.7156838186993649, "min_ratio": 0.7156838186993649, "max_ratio": 0.7156838186993649}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.7144358817486884, "min_ratio": 0.7144358817486884, "max_ratio": 0.7144358817486884}
- s_output_accumulator_512_1024_512 vs c_plain_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.9588743275570113, "min_ratio": 1.9588743275570113, "max_ratio": 1.9588743275570113}
- s_plain_512_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.9550502371054148, "min_ratio": 1.9550502371054148, "max_ratio": 1.9550502371054148}
- s_interleaved_output_accumulator_512_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.3195498619826138, "min_ratio": 1.3195498619826138, "max_ratio": 1.3195498619826138}
- s_interleaved_2048_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.8847606183735304, "min_ratio": 0.6887587896528683, "max_ratio": 1.0807624470941923}
- s_output_accumulator_1024_1024_1024 vs c_plain_128_512_512: {"counts": {"win": 3}, "median_ratio": 1.2508234964965914, "min_ratio": 1.1327630988615494, "max_ratio": 2.1515496675791943}
- s_plain_1024_1024_1024 vs c_plain_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.4671722985799982, "min_ratio": 1.4671722985799982, "max_ratio": 1.4671722985799982}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_1024_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.7110650024499012, "min_ratio": 0.7110650024499012, "max_ratio": 0.7110650024499012}
- s_output_accumulator_1024_1024_512 vs c_plain_512_512_256: {"counts": {"loss": 2, "win": 1}, "median_ratio": 0.8639372306278752, "min_ratio": 0.7412612908910917, "max_ratio": 1.346639772035988}
- s_plain_512_1024_512 vs c_output_accumulator_128_512_512: {"counts": {"win": 1}, "median_ratio": 1.4343485093458654, "min_ratio": 1.4343485093458654, "max_ratio": 1.4343485093458654}
- s_plain_512_1024_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.9603843364209189, "min_ratio": 0.9603843364209189, "max_ratio": 0.9603843364209189}
- s_output_accumulator_1024_1024_512 vs c_plain_1024_1024_1024: {"counts": {"inconclusive": 1}, "median_ratio": 1.0011631786599051, "min_ratio": 1.0011631786599051, "max_ratio": 1.0011631786599051}
- s_interleaved_2048_1024_512 vs c_plain_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0071914576988872, "min_ratio": 1.0071914576988872, "max_ratio": 1.0071914576988872}
- s_interleaved_output_accumulator_2048_2048_512 vs c_output_accumulator_512_512_256: {"counts": {"win": 1, "loss": 1}, "median_ratio": 0.7777975485130629, "min_ratio": 0.4503701227809976, "max_ratio": 1.1052249742451283}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 2}, "median_ratio": 1.0461181089050098, "min_ratio": 1.0291520650053823, "max_ratio": 1.063084152804637}
- s_interleaved_1024_1024_512 vs c_plain_512_512_256: {"counts": {"win": 3}, "median_ratio": 1.3018353703924737, "min_ratio": 1.1345900420520123, "max_ratio": 1.5222229203969253}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_plain_2048_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0229782505596592, "min_ratio": 1.0229782505596592, "max_ratio": 1.0229782505596592}
- s_interleaved_512_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"inconclusive": 1}, "median_ratio": 0.9846168299320698, "min_ratio": 0.9846168299320698, "max_ratio": 0.9846168299320698}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_512_1024_512: {"counts": {"win": 1}, "median_ratio": 1.039296698176415, "min_ratio": 1.039296698176415, "max_ratio": 1.039296698176415}
- s_interleaved_1024_1024_1024 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 1.506673897041455, "min_ratio": 1.506673897041455, "max_ratio": 1.506673897041455}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_512_1024_1024: {"counts": {"inconclusive": 2}, "median_ratio": 1.0361381170044783, "min_ratio": 1.0069627871721116, "max_ratio": 1.0653134468368453}
- s_plain_1024_1024_1024 vs c_output_accumulator_512_1024_1024: {"counts": {"inconclusive": 1, "win": 2}, "median_ratio": 1.0627858775815746, "min_ratio": 1.017934080586691, "max_ratio": 1.0743207572067828}
- s_plain_1024_1024_512 vs c_plain_1024_2048_512: {"counts": {"inconclusive": 2}, "median_ratio": 1.041996842986031, "min_ratio": 1.0228822505673696, "max_ratio": 1.0611114354046924}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"win": 1}, "median_ratio": 1.038309531647838, "min_ratio": 1.038309531647838, "max_ratio": 1.038309531647838}
- s_interleaved_output_accumulator_512_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"inconclusive": 1, "loss": 3}, "median_ratio": 0.9556654314320219, "min_ratio": 0.9258782589806375, "max_ratio": 1.0070898111162654}
- c_plain_1024_512_512 vs native_xla: {"counts": {"inconclusive": 1}, "median_ratio": 0.9829307743751651, "min_ratio": 0.9829307743751651, "max_ratio": 0.9829307743751651}
- s_output_accumulator_2048_2048_512 vs c_plain_1024_512_512: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.9510380237265195, "min_ratio": 0.7724687926968861, "max_ratio": 1.1296072547561529}
- s_output_accumulator_2048_2048_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.759283348586199, "min_ratio": 0.759283348586199, "max_ratio": 0.759283348586199}
- native_xla vs c_plain_1024_512_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0173656437155358, "min_ratio": 1.0173656437155358, "max_ratio": 1.0173656437155358}
- s_plain_1024_1024_1024 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.9344807080916054, "min_ratio": 0.9344807080916054, "max_ratio": 0.9344807080916054}
- s_plain_512_1024_512 vs c_plain_1024_1024_1024: {"counts": {"inconclusive": 1, "loss": 1, "win": 1}, "median_ratio": 1.0143278600495496, "min_ratio": 0.9176215227723915, "max_ratio": 2.585056895991262}
- s_interleaved_1024_1024_512 vs c_plain_512_1024_512: {"counts": {"win": 2}, "median_ratio": 1.0814527000957308, "min_ratio": 1.031011265811408, "max_ratio": 1.1318941343800535}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_512_1024_1024: {"counts": {"inconclusive": 2}, "median_ratio": 0.9987389836244254, "min_ratio": 0.9964263053636282, "max_ratio": 1.0010516618852225}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.9382003560293213, "min_ratio": 0.9382003560293213, "max_ratio": 0.9382003560293213}
- s_interleaved_2048_1024_512 vs c_plain_512_512_256: {"counts": {"loss": 1}, "median_ratio": 0.7945170225893327, "min_ratio": 0.7945170225893327, "max_ratio": 0.7945170225893327}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.3392523238420708, "min_ratio": 1.3392523238420708, "max_ratio": 1.3392523238420708}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 1.1591502772362923, "min_ratio": 1.1591502772362923, "max_ratio": 1.1591502772362923}
- s_output_accumulator_1024_1024_512 vs c_plain_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0021719380765033, "min_ratio": 1.0021719380765033, "max_ratio": 1.0021719380765033}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"win": 1, "inconclusive": 1}, "median_ratio": 1.0256956445207006, "min_ratio": 1.0061019198425891, "max_ratio": 1.0452893691988119}
- s_interleaved_512_1024_512 vs c_plain_1024_2048_512: {"counts": {"inconclusive": 1, "loss": 1}, "median_ratio": 0.9670845831850775, "min_ratio": 0.9215048765515632, "max_ratio": 1.0126642898185918}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.075263165496953, "min_ratio": 1.075263165496953, "max_ratio": 1.075263165496953}
- s_interleaved_output_accumulator_2048_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"loss": 2}, "median_ratio": 0.5330488041402487, "min_ratio": 0.2921467086081525, "max_ratio": 0.7739508996723449}
- s_plain_2048_1024_512 vs c_plain_512_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0818483363315619, "min_ratio": 1.0818483363315619, "max_ratio": 1.0818483363315619}
- s_plain_2048_1024_512 vs native_xla: {"counts": {"win": 2}, "median_ratio": 1.0570325193716557, "min_ratio": 1.0251913385969564, "max_ratio": 1.0888737001463549}
- s_interleaved_output_accumulator_2048_2048_512 vs c_plain_2048_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.021735139787202, "min_ratio": 1.021735139787202, "max_ratio": 1.021735139787202}
- s_plain_1024_1024_1024 vs c_plain_1024_512_512: {"counts": {"win": 1}, "median_ratio": 1.0615261083264391, "min_ratio": 1.0615261083264391, "max_ratio": 1.0615261083264391}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 2}, "median_ratio": 1.0379953192769935, "min_ratio": 1.0375843177627306, "max_ratio": 1.0384063207912564}
- s_interleaved_2048_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.0995054330854537, "min_ratio": 1.0995054330854537, "max_ratio": 1.0995054330854537}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_128_512_512: {"counts": {"win": 1}, "median_ratio": 2.0516266855630354, "min_ratio": 2.0516266855630354, "max_ratio": 2.0516266855630354}
- s_interleaved_1024_1024_1024 vs c_plain_1024_1024_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0537019201081925, "min_ratio": 1.0134072489199413, "max_ratio": 1.0939965912964436}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.343342314919805, "min_ratio": 1.343342314919805, "max_ratio": 1.343342314919805}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"win": 2}, "median_ratio": 1.361923432056774, "min_ratio": 1.2925896899223575, "max_ratio": 1.4312571741911901}
- s_output_accumulator_2048_2048_512 vs c_plain_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.0428930411948838, "min_ratio": 1.0428930411948838, "max_ratio": 1.0428930411948838}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_1024_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 0.9797158015237996, "min_ratio": 0.9797158015237996, "max_ratio": 0.9797158015237996}
- c_plain_1024_1024_1024 vs native_xla: {"counts": {"loss": 2}, "median_ratio": 0.7151416677584352, "min_ratio": 0.4754340685342087, "max_ratio": 0.9548492669826618}
- s_interleaved_1024_1024_1024 vs c_plain_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.0205600284397331, "min_ratio": 1.0205600284397331, "max_ratio": 1.0205600284397331}
- native_xla vs c_plain_1024_1024_1024: {"counts": {"win": 2}, "median_ratio": 1.5753133913690034, "min_ratio": 1.047285717839021, "max_ratio": 2.1033410648989856}
- s_interleaved_512_1024_512 vs c_plain_128_512_512: {"counts": {"win": 2}, "median_ratio": 2.270899155200329, "min_ratio": 1.899695643685815, "max_ratio": 2.6421026667148433}
- s_plain_512_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.179452128460702, "min_ratio": 1.179452128460702, "max_ratio": 1.179452128460702}
- s_plain_1024_1024_512 vs c_plain_2048_1024_512: {"counts": {"inconclusive": 2}, "median_ratio": 0.9966062238703199, "min_ratio": 0.9965516959380901, "max_ratio": 0.9966607518025499}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_2048_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0119312950659247, "min_ratio": 1.0119312950659247, "max_ratio": 1.0119312950659247}
- s_interleaved_2048_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0076880378086956, "min_ratio": 1.0076880378086956, "max_ratio": 1.0076880378086956}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0781163734776722, "min_ratio": 1.0781163734776722, "max_ratio": 1.0781163734776722}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_512_1024_512: {"counts": {"inconclusive": 1}, "median_ratio": 1.0078725780683524, "min_ratio": 1.0078725780683524, "max_ratio": 1.0078725780683524}
- s_interleaved_output_accumulator_2048_2048_512 vs c_output_accumulator_1024_512_512: {"counts": {"win": 2}, "median_ratio": 1.1822591927916468, "min_ratio": 1.1388410114710794, "max_ratio": 1.2256773741122144}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_128_512_512: {"counts": {"win": 1}, "median_ratio": 2.0321937328093025, "min_ratio": 2.0321937328093025, "max_ratio": 2.0321937328093025}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 2.0951511158182408, "min_ratio": 2.0951511158182408, "max_ratio": 2.0951511158182408}
- s_interleaved_1024_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.9557349941174565, "min_ratio": 0.9557349941174565, "max_ratio": 0.9557349941174565}
- s_plain_1024_1024_1024 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.3549779349973383, "min_ratio": 1.3549779349973383, "max_ratio": 1.3549779349973383}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"win": 1, "loss": 1}, "median_ratio": 0.830654817032759, "min_ratio": 0.5519298051266124, "max_ratio": 1.1093798289389056}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.9129621608439485, "min_ratio": 0.9129621608439485, "max_ratio": 0.9129621608439485}
- s_interleaved_output_accumulator_2048_2048_512 vs c_plain_512_512_256: {"counts": {"win": 1, "loss": 1}, "median_ratio": 0.9546029003013803, "min_ratio": 0.4345457861888948, "max_ratio": 1.4746600144138657}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_plain_1024_1024_1024: {"counts": {"win": 2}, "median_ratio": 1.0509959320465743, "min_ratio": 1.042688852098097, "max_ratio": 1.0593030119950515}
- s_interleaved_1024_1024_512 vs c_output_accumulator_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0175089487765532, "min_ratio": 1.0175089487765532, "max_ratio": 1.0175089487765532}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.0570722203106333, "min_ratio": 1.0570722203106333, "max_ratio": 1.0570722203106333}
- s_output_accumulator_1024_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"inconclusive": 1, "win": 1}, "median_ratio": 1.0100262745900728, "min_ratio": 1.0057309353561907, "max_ratio": 1.014321613823955}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_1024_512_512: {"counts": {"win": 2}, "median_ratio": 1.1510639994674796, "min_ratio": 1.1073021098703006, "max_ratio": 1.1948258890646588}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"inconclusive": 1}, "median_ratio": 0.9882101224395968, "min_ratio": 0.9882101224395968, "max_ratio": 0.9882101224395968}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_512_1024_512: {"counts": {"win": 1}, "median_ratio": 1.1436651626615786, "min_ratio": 1.1436651626615786, "max_ratio": 1.1436651626615786}
- c_output_accumulator_1024_1024_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.5034177644956075, "min_ratio": 0.5034177644956075, "max_ratio": 0.5034177644956075}
- native_xla vs c_output_accumulator_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.9864217564947004, "min_ratio": 1.9864217564947004, "max_ratio": 1.9864217564947004}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.017148051320656, "min_ratio": 1.017148051320656, "max_ratio": 1.017148051320656}
- s_interleaved_1024_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.5661000859291048, "min_ratio": 0.5661000859291048, "max_ratio": 0.5661000859291048}
- s_interleaved_512_1024_512 vs c_plain_512_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.748339652758673, "min_ratio": 1.748339652758673, "max_ratio": 1.748339652758673}
- s_interleaved_1024_1024_1024 vs c_plain_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.1915812931005625, "min_ratio": 1.1915812931005625, "max_ratio": 1.1915812931005625}
- s_interleaved_output_accumulator_2048_2048_512 vs c_plain_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.6472952513078178, "min_ratio": 0.6472952513078178, "max_ratio": 0.6472952513078178}
- s_output_accumulator_1024_1024_512 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 1.2233181121388543, "min_ratio": 1.2233181121388543, "max_ratio": 1.2233181121388543}
- s_output_accumulator_2048_1024_512 vs c_plain_1024_512_512: {"counts": {"loss": 1}, "median_ratio": 0.6490070090927601, "min_ratio": 0.6490070090927601, "max_ratio": 0.6490070090927601}
- s_interleaved_2048_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.040278668959838, "min_ratio": 1.040278668959838, "max_ratio": 1.040278668959838}
- s_interleaved_output_accumulator_512_1024_512 vs c_output_accumulator_1024_512_512: {"counts": {"win": 1}, "median_ratio": 1.8649225241410285, "min_ratio": 1.8649225241410285, "max_ratio": 1.8649225241410285}
- s_plain_1024_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.484453515328829, "min_ratio": 1.484453515328829, "max_ratio": 1.484453515328829}
- s_plain_1024_1024_1024 vs c_output_accumulator_128_512_512: {"counts": {"loss": 1, "win": 1}, "median_ratio": 1.0270156054458064, "min_ratio": 0.8062710624455555, "max_ratio": 1.2477601484460574}
- s_output_accumulator_1024_1024_1024 vs c_plain_1024_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.7013576142682129, "min_ratio": 0.7013576142682129, "max_ratio": 0.7013576142682129}
- c_output_accumulator_512_1024_1024 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.9292422371836824, "min_ratio": 0.9292422371836824, "max_ratio": 0.9292422371836824}
- s_interleaved_1024_1024_512 vs c_output_accumulator_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.552409089617124, "min_ratio": 0.552409089617124, "max_ratio": 0.552409089617124}
- s_interleaved_1024_1024_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.5133218582764175, "min_ratio": 0.5133218582764175, "max_ratio": 0.5133218582764175}
- native_xla vs c_output_accumulator_512_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.07614565931782, "min_ratio": 1.07614565931782, "max_ratio": 1.07614565931782}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.800531098365124, "min_ratio": 1.800531098365124, "max_ratio": 1.800531098365124}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 3.056239466944099, "min_ratio": 3.056239466944099, "max_ratio": 3.056239466944099}
- s_output_accumulator_2048_2048_512 vs c_plain_2048_1024_512: {"counts": {"win": 2}, "median_ratio": 1.1079230847370463, "min_ratio": 1.100113625196844, "max_ratio": 1.1157325442772488}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.834157617641969, "min_ratio": 1.834157617641969, "max_ratio": 1.834157617641969}
- s_interleaved_output_accumulator_2048_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.33648663212885754, "min_ratio": 0.33648663212885754, "max_ratio": 0.33648663212885754}
- s_interleaved_output_accumulator_1024_1024_1024 vs native_xla: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.7856190880574023, "min_ratio": 0.5036287408033491, "max_ratio": 1.0676094353114556}
- s_output_accumulator_2048_1024_512 vs c_output_accumulator_2048_2048_512: {"counts": {"win": 1}, "median_ratio": 1.0270909454303514, "min_ratio": 1.0270909454303514, "max_ratio": 1.0270909454303514}
- s_output_accumulator_2048_2048_512 vs c_output_accumulator_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.590854217753332, "min_ratio": 0.590854217753332, "max_ratio": 0.590854217753332}
- s_interleaved_2048_1024_512 vs c_plain_1024_512_512: {"counts": {"loss": 1, "win": 1}, "median_ratio": 0.8839138522858632, "min_ratio": 0.6171952040454379, "max_ratio": 1.1506325005262883}
- s_output_accumulator_512_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.8704668763025594, "min_ratio": 1.8704668763025594, "max_ratio": 1.8704668763025594}
- s_plain_1024_1024_512 vs c_plain_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0313784259176229, "min_ratio": 1.0313784259176229, "max_ratio": 1.0313784259176229}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_512_512_256: {"counts": {"loss": 1}, "median_ratio": 0.7519169329073482, "min_ratio": 0.7519169329073482, "max_ratio": 0.7519169329073482}
- s_interleaved_512_1024_512 vs c_output_accumulator_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.9481095964108923, "min_ratio": 1.9481095964108923, "max_ratio": 1.9481095964108923}
- s_plain_2048_1024_512 vs c_output_accumulator_128_512_512: {"counts": {"loss": 1}, "median_ratio": 0.7163622249255446, "min_ratio": 0.7163622249255446, "max_ratio": 0.7163622249255446}
- s_output_accumulator_1024_1024_1024 vs c_output_accumulator_1024_512_512: {"counts": {"win": 1}, "median_ratio": 1.109090263639226, "min_ratio": 1.109090263639226, "max_ratio": 1.109090263639226}
- s_plain_512_1024_512 vs c_plain_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.8783989366212281, "min_ratio": 1.8783989366212281, "max_ratio": 1.8783989366212281}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 2.4566206650763025, "min_ratio": 2.4566206650763025, "max_ratio": 2.4566206650763025}
- c_output_accumulator_1024_2048_512 vs native_xla: {"counts": {"win": 1}, "median_ratio": 1.0185834925734545, "min_ratio": 1.0185834925734545, "max_ratio": 1.0185834925734545}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_output_accumulator_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.0481314915227389, "min_ratio": 1.0481314915227389, "max_ratio": 1.0481314915227389}
- native_xla vs c_output_accumulator_1024_2048_512: {"counts": {"loss": 1}, "median_ratio": 0.9817555529723899, "min_ratio": 0.9817555529723899, "max_ratio": 0.9817555529723899}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_2048_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0560780278023376, "min_ratio": 1.0560780278023376, "max_ratio": 1.0560780278023376}
- s_plain_512_1024_512 vs c_plain_128_512_512: {"counts": {"win": 1}, "median_ratio": 2.4402097419264486, "min_ratio": 2.4402097419264486, "max_ratio": 2.4402097419264486}
- s_interleaved_output_accumulator_512_1024_512 vs c_plain_2048_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.9668399403121628, "min_ratio": 0.9668399403121628, "max_ratio": 0.9668399403121628}
- s_interleaved_output_accumulator_1024_1024_512 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.5205288956550422, "min_ratio": 1.5205288956550422, "max_ratio": 1.5205288956550422}
- s_output_accumulator_2048_1024_512 vs c_plain_1024_1024_512: {"counts": {"win": 1}, "median_ratio": 1.0966548972319354, "min_ratio": 1.0966548972319354, "max_ratio": 1.0966548972319354}
- s_interleaved_512_1024_512 vs c_plain_1024_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.9436224256188732, "min_ratio": 0.9436224256188732, "max_ratio": 0.9436224256188732}
- s_plain_2048_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.039697292026748, "min_ratio": 1.039697292026748, "max_ratio": 1.039697292026748}
- s_interleaved_output_accumulator_2048_1024_512 vs c_plain_512_1024_512: {"counts": {"win": 1}, "median_ratio": 1.1699920826392551, "min_ratio": 1.1699920826392551, "max_ratio": 1.1699920826392551}
- s_interleaved_output_accumulator_512_1024_512 vs native_xla: {"counts": {"loss": 1}, "median_ratio": 0.8897696523091057, "min_ratio": 0.8897696523091057, "max_ratio": 0.8897696523091057}
- s_interleaved_output_accumulator_1024_1024_1024 vs c_output_accumulator_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.6452314348072292, "min_ratio": 1.6452314348072292, "max_ratio": 1.6452314348072292}
- s_plain_512_1024_512 vs c_output_accumulator_512_1024_512: {"counts": {"loss": 1}, "median_ratio": 0.9872971637202329, "min_ratio": 0.9872971637202329, "max_ratio": 0.9872971637202329}
- s_plain_2048_1024_512 vs c_plain_512_512_256: {"counts": {"win": 1}, "median_ratio": 1.6449958485764453, "min_ratio": 1.6449958485764453, "max_ratio": 1.6449958485764453}
- s_interleaved_2048_1024_512 vs c_output_accumulator_1024_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.04394805817002, "min_ratio": 1.04394805817002, "max_ratio": 1.04394805817002}
- s_output_accumulator_512_1024_512 vs c_plain_512_1024_1024: {"counts": {"loss": 1}, "median_ratio": 0.9755300337155467, "min_ratio": 0.9755300337155467, "max_ratio": 0.9755300337155467}
- s_interleaved_1024_1024_1024 vs c_output_accumulator_512_1024_1024: {"counts": {"win": 1}, "median_ratio": 1.1101034872143303, "min_ratio": 1.1101034872143303, "max_ratio": 1.1101034872143303}
- s_plain_1024_1024_1024 vs c_plain_1024_2048_512: {"counts": {"win": 1}, "median_ratio": 1.044111096504831, "min_ratio": 1.044111096504831, "max_ratio": 1.044111096504831}

## 20260919T063644Z-N5-confirm-v5e-v004-155d9f

Archive integrity: PASS

Status counts: `{"ok": 96}`

- native_xla vs cubic_selected: {"counts": {"win": 10, "inconclusive": 3, "loss": 3}, "median_ratio": 1.0202402805384592, "min_ratio": 0.9023955538927131, "max_ratio": 1.1185941416484506}
- cubic_selected vs native_xla: {"counts": {"loss": 10, "inconclusive": 3, "win": 3}, "median_ratio": 0.980177342226908, "min_ratio": 0.8939792930850856, "max_ratio": 1.1081614882589406}
- strassen_selected vs native_xla: {"counts": {"inconclusive": 4, "loss": 6, "win": 6}, "median_ratio": 0.9984912605226889, "min_ratio": 0.8954705488013018, "max_ratio": 1.124206252870119}
- strassen_selected vs cubic_selected: {"counts": {"win": 6, "inconclusive": 8, "loss": 2}, "median_ratio": 1.0035076353806862, "min_ratio": 0.956902170729489, "max_ratio": 1.0904757575276238}

## 20260919T063959Z-N6-v5e-v004-d93a61

Archive integrity: PASS

Status counts: `{"ok": 18}`

- native_xla vs cubic_selected: {"counts": {"win": 2, "loss": 3, "inconclusive": 1}, "median_ratio": 0.9505709792066737, "min_ratio": 0.8908436217609473, "max_ratio": 1.0195496790373373}
- cubic_selected vs native_xla: {"counts": {"loss": 2, "win": 3, "inconclusive": 1}, "median_ratio": 1.05354235683862, "min_ratio": 0.9808251824905716, "max_ratio": 1.1225314696908097}
- strassen_selected vs native_xla: {"counts": {"win": 4, "loss": 1, "inconclusive": 1}, "median_ratio": 1.0214540118283895, "min_ratio": 0.9190244636536563, "max_ratio": 1.122228085931586}
- strassen_selected vs cubic_selected: {"counts": {"win": 2, "inconclusive": 2, "loss": 2}, "median_ratio": 0.9953489924101535, "min_ratio": 0.9055113485644197, "max_ratio": 1.0397671390489494}

## 20260919T064743Z-N6-v5e-v004-e7bc5e

Archive integrity: PASS

Status counts: `{"ok": 18}`

- native_xla vs cubic_selected: {"counts": {"win": 2, "loss": 2, "inconclusive": 2}, "median_ratio": 0.9845846762287416, "min_ratio": 0.8800682025573628, "max_ratio": 1.0242029509543702}
- cubic_selected vs native_xla: {"counts": {"loss": 2, "win": 2, "inconclusive": 2}, "median_ratio": 1.016174102439702, "min_ratio": 0.9763689892400549, "max_ratio": 1.1362755717047055}
- strassen_selected vs native_xla: {"counts": {"win": 4, "inconclusive": 2}, "median_ratio": 1.0242438356151116, "min_ratio": 0.9570345579216385, "max_ratio": 1.1213637342241909}
- strassen_selected vs cubic_selected: {"counts": {"win": 2, "inconclusive": 3, "loss": 1}, "median_ratio": 0.9868802322856498, "min_ratio": 0.9356775971592227, "max_ratio": 1.0453139285880857}

## 20260919T064008Z-N6a-selector-fit-v002-6ef3f3

Archive integrity: PASS

Status counts: `{}`


