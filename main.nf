#!/usr/bin/env nextflow
/*
 * OligoGym benchmark calibration pipeline.
 *
 * Phase 2: measure real per-model-class cost on the actual AWS Batch hardware
 * (c6id/r6id/m6id for cpu-class configs, g4dn/T4 for gpu-class).
 *
 * Three independent measurement tracks, selectable with --mode:
 *   calibrate  one instrumented run per config (wall/RSS/VRAM/epochs per fold)
 *   cache      the same configs run twice, with and without the feature cache
 *   pack       GPU packing sweep: 1/2/4/8 concurrent trainers on one T4
 */

nextflow.enable.dsl = 2

params.mode        = 'calibrate'
params.configs     = "${projectDir}/configs/*.yaml"
params.manifest    = "${projectDir}/cal_manifest.csv"
params.outdir      = 'results'
params.seed        = 0
params.folds       = 5
params.pack_levels = '1 2 4 8'
params.pack_reps   = 2
params.pack_model  = 'CNN'
params.pack_dataset = 'siRNA1'
// 'cpu' | 'gpu' | 'all' -- routes configs to the matching Batch env so the CPU and
// GPU arms can run concurrently on their own queues.
params.compute_class = 'all'

// ---------------------------------------------------------------- calibrate
process CALIBRATE_CPU {
    tag "${cfg.simpleName}"
    label 'measure_cpu'
    errorStrategy 'ignore'

    input:
    tuple val(meta), path(cfg)

    output:
    path "${cfg.simpleName}.jsonl", emit: recs, optional: true
    path "${cfg.simpleName}.log",   emit: logs, optional: true

    script:
    """
    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}

    python /opt/oligogym-bench/calibrate.py \\
        --config ${cfg} --out ${cfg.simpleName}.jsonl \\
        --seed ${params.seed} --folds ${params.folds} \\
        --tag "${meta.compute_class}|${meta.model}|${meta.dataset}|${meta.tier}" \\
        > ${cfg.simpleName}.log 2>&1

    echo "=====RESULTS_BEGIN====="
    cat ${cfg.simpleName}.jsonl
    echo "=====RESULTS_END====="
    """
}

process CALIBRATE_GPU {
    tag "${cfg.simpleName}"
    label 'measure_gpu'
    errorStrategy 'ignore'

    input:
    tuple val(meta), path(cfg)

    output:
    path "${cfg.simpleName}.jsonl", emit: recs, optional: true
    path "${cfg.simpleName}.log",   emit: logs, optional: true

    script:
    """
    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}
    nvidia-smi > ${cfg.simpleName}.log 2>&1 || echo "no nvidia-smi" > ${cfg.simpleName}.log

    python /opt/oligogym-bench/calibrate.py \\
        --config ${cfg} --out ${cfg.simpleName}.jsonl \\
        --seed ${params.seed} --folds ${params.folds} \\
        --tag "${meta.compute_class}|${meta.model}|${meta.dataset}|${meta.tier}" \\
        >> ${cfg.simpleName}.log 2>&1

    echo "=====RESULTS_BEGIN====="
    cat ${cfg.simpleName}.jsonl
    echo "=====RESULTS_END====="
    """
}

// ---------------------------------------------------------------- feature cache
// Same config, run with --cache and without, so the saving is a paired measurement
// on identical folds rather than a comparison across two different runs.
process CACHE_AB {
    tag "${cfg.simpleName}"
    label 'measure_cpu'
    errorStrategy 'ignore'

    input:
    tuple val(meta), path(cfg)

    output:
    path "${cfg.simpleName}_cache.jsonl", emit: recs, optional: true

    script:
    """
    export OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}

    python /opt/oligogym-bench/calibrate.py --config ${cfg} \\
        --out ${cfg.simpleName}_cache.jsonl --seed ${params.seed} \\
        --folds ${params.folds} --tag "nocache|${meta.model}|${meta.dataset}" \\
        > nocache.log 2>&1

    python /opt/oligogym-bench/calibrate.py --config ${cfg} --cache \\
        --out ${cfg.simpleName}_cache.jsonl --seed ${params.seed} \\
        --folds ${params.folds} --tag "cache|${meta.model}|${meta.dataset}" \\
        > cache.log 2>&1
    """
}

// ---------------------------------------------------------------- GPU packing
process GPU_PACK {
    tag "${model}:${dataset}"
    label 'gpu_pack'

    input:
    tuple val(model), val(dataset)

    output:
    path "gpu_packing_${model}_${dataset}.jsonl", emit: recs

    script:
    """
    nvidia-smi > gpu_info.txt 2>&1 || echo "no nvidia-smi" > gpu_info.txt

    python /opt/oligogym-bench/gpu_pack.py \\
        --levels ${params.pack_levels} \\
        --reps ${params.pack_reps} \\
        --folds 1 \\
        --model ${model} \\
        --dataset ${dataset} \\
        --threads-per-proc 1 \\
        --out gpu_packing_${model}_${dataset}.jsonl

    echo "=====RESULTS_BEGIN====="
    cat gpu_packing_${model}_${dataset}.jsonl
    echo "=====RESULTS_END====="
    echo "=====GPUINFO_BEGIN====="
    cat gpu_info.txt
    echo "=====GPUINFO_END====="
    """
}

// ---------------------------------------------------------------- image validation
process VALIDATE_IMAGE_CPU {
    label 'validate_cpu'
    publishDir "${params.outdir}", mode: 'copy'
    errorStrategy 'ignore'

    output:
    path "validation_cpu.txt",  emit: log
    path "validation_cpu.json", emit: report, optional: true

    script:
    """
    echo "== CPU-class task ==" > validation_cpu.txt
    python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'arch', torch.cuda.get_arch_list())" >> validation_cpu.txt 2>&1
    python /opt/oligogym-bench/validate_image.py --build-time \\
        --json-out validation_cpu.json >> validation_cpu.txt 2>&1 || true
    """
}

process VALIDATE_IMAGE_GPU {
    label 'validate_gpu'
    // The report is the deliverable even when a check FAILS -- publish it either
    // way, otherwise a failing check destroys the evidence of what failed.
    publishDir "${params.outdir}", mode: 'copy'
    errorStrategy 'ignore'

    output:
    path "validation_gpu.txt",  emit: log
    path "validation_gpu.json", emit: report, optional: true

    script:
    """
    echo "== GPU-class task ==" > validation_gpu.txt
    nvidia-smi >> validation_gpu.txt 2>&1 || echo "no nvidia-smi" >> validation_gpu.txt
    python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, 'arch', torch.cuda.get_arch_list())" >> validation_gpu.txt 2>&1
    # `|| true`: a failed check must still leave the log behind to be published.
    python /opt/oligogym-bench/validate_image.py --gpu \\
        --json-out validation_gpu.json >> validation_gpu.txt 2>&1 || true
    echo "exit_recorded=$?" >> validation_gpu.txt
    """
}

process COLLECT {
    label 'tiny'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path 'part_*.jsonl'
    val  name

    output:
    path "${name}.jsonl"

    script:
    """
    cat part_*.jsonl > ${name}.jsonl
    wc -l ${name}.jsonl
    echo "=====COLLECTED_BEGIN====="
    cat ${name}.jsonl
    echo "=====COLLECTED_END====="
    """
}

// ---------------------------------------------------------------- workflow
workflow {
    if (params.mode == 'validate') {
        // GPU checks need a g4dn task; the CPU-class check confirms the same image
        // runs on c6id/r6id/m6id. Launch this mode on each env separately.
        if (params.compute_class in ['all', 'gpu']) {
            VALIDATE_IMAGE_GPU()
            VALIDATE_IMAGE_GPU.out.log
                .collectFile(name: 'validation_gpu.txt', storeDir: params.outdir)
            VALIDATE_IMAGE_GPU.out.report
                .collectFile(name: 'validation_gpu.json', storeDir: params.outdir)
        }
        if (params.compute_class in ['all', 'cpu']) {
            VALIDATE_IMAGE_CPU()
            VALIDATE_IMAGE_CPU.out.log
                .collectFile(name: 'validation_cpu.txt', storeDir: params.outdir)
            VALIDATE_IMAGE_CPU.out.report
                .collectFile(name: 'validation_cpu.json', storeDir: params.outdir)
        }
    }
    else if (params.mode == 'pack') {
        // Sweep concurrency for the model classes whose cost profile differs most:
        // CNN (cheap, dataloader-bound), Transformer (compute-bound),
        // RNAFM_Transformer (the one genuine VRAM consumer).
        Channel.of(
            ['CNN', params.pack_dataset],
            ['Transformer', params.pack_dataset],
            ['RNAFM_Transformer', params.pack_dataset],
        ) | GPU_PACK
        COLLECT(GPU_PACK.out.recs.collect(), Channel.value('gpu_packing'))
    }
    else {
        // meta from the manifest, joined to the config files by name
        meta_ch = Channel
            .fromPath(params.manifest)
            .splitCsv(header: true)
            .map { row -> [row.name, row] }

        cfg_ch = Channel
            .fromPath(params.configs)
            .map { f -> [f.simpleName, f] }

        joined = meta_ch.join(cfg_ch).map { _n, row, f -> [row, f] }

        if (params.compute_class != 'all') {
            joined = joined.filter { m, _f -> m.compute_class == params.compute_class }
        }

        if (params.mode == 'cache') {
            // Featurization cost is what matters here, and it is dataset-scale
            // driven, so restrict to the featurizer/model pairs that dominate.
            CACHE_AB(joined.filter { m, _f ->
                m.featurizer in ['OneHotEncoder', 'KMersCounts', 'RNAFMEmbeddings']
            })
            COLLECT(CACHE_AB.out.recs.collect(), Channel.value('cache_ab'))
        } else {
            branched = joined.branch {
                gpu: it[0].compute_class == 'gpu'
                cpu: true
            }
            CALIBRATE_CPU(branched.cpu)
            CALIBRATE_GPU(branched.gpu)
            recs = CALIBRATE_CPU.out.recs.mix(CALIBRATE_GPU.out.recs)
            COLLECT(recs.collect(), Channel.value('calibration'))
            CALIBRATE_CPU.out.logs.mix(CALIBRATE_GPU.out.logs)
                .collectFile(name: 'calibration_logs.txt', storeDir: params.outdir)
        }
    }
}
