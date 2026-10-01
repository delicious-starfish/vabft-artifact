"""V-ABFT FPR test on the CLEAN LLM dataset (checksum_llm_data, not data1).

Loads all (a, b) matrix pairs from `$LLM_DATA_ROOT/{atte_c,atte_r,mlp_r}/` (set LLM_DATA_ROOT)
and runs `utils.FT_matmul` (default = my_bound_improve_robust = V-ABFT).

A "failure" here = false positive, since these are real LLM matrices with no injected error.
Outputs per-folder counts and overall FPR.
"""

import os
import glob
import torch
import torch_npu
from datetime import datetime

import utils

device = torch.device('npu:0' if torch_npu.npu.is_available() else 'cpu')

DATA_ROOT = os.environ.get('LLM_DATA_ROOT', './checksum_llm_data')
SUBDIRS = ['atte_c', 'atte_r', 'mlp_r']
LOG_PATH = os.path.join(os.environ.get('VABFT_ROOT', '.'), 'results_phase2', 'E1_fpr_clean.log')

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def run_folder(folder: str) -> tuple[int, int, list[str]]:
    """Returns (total_pairs, false_positive_count, failure_filenames)."""
    pattern = os.path.join(folder, '*_a_*.pth')
    a_files = sorted(glob.glob(pattern))
    fp_count = 0
    failures = []

    for path_a in a_files:
        name_a = os.path.basename(path_a)
        # name like rank_X_dw_step_Y_mb_Z_layer_W_a_NNN.pth
        # corresponding b: replace "_a_" -> "_b_"
        path_b = path_a.replace('_a_', '_b_', 1)
        if not os.path.exists(path_b):
            failures.append(f'{name_a} (b missing)')
            continue
        try:
            a = torch.load(path_a, map_location='cpu').to(device)
            b = torch.load(path_b, map_location='cpu').to(device)
            success, _ = utils.FT_matmul(a, b)
            if not success:
                fp_count += 1
                failures.append(name_a)
        except Exception as e:
            failures.append(f'{name_a} (exception: {type(e).__name__}: {str(e)[:80]})')

    return len(a_files), fp_count, failures


def main():
    print(f'[{datetime.now()}] V-ABFT FPR test on clean LLM dataset')
    print(f'Data root: {DATA_ROOT}')
    print(f'Algorithm: utils.FT_matmul (default = my_bound_improve_robust)')
    print('=' * 80)

    total_all = 0
    fp_all = 0
    summary = []

    with open(LOG_PATH, 'w') as logf:
        logf.write(f'V-ABFT FPR Test (clean dataset)\n')
        logf.write(f'Started: {datetime.now()}\n')
        logf.write(f'Data root: {DATA_ROOT}\n')
        logf.write('=' * 80 + '\n\n')

        for sub in SUBDIRS:
            folder = os.path.join(DATA_ROOT, sub)
            if not os.path.exists(folder):
                print(f'[SKIP] {folder} does not exist')
                logf.write(f'[SKIP] {folder} does not exist\n\n')
                continue

            print(f'\n--- {sub} ---')
            logf.write(f'--- {sub} ---\n')

            total, fp, failures = run_folder(folder)
            fpr = (fp / total * 100) if total > 0 else 0.0

            summary.append((sub, total, fp, fpr))
            total_all += total
            fp_all += fp

            print(f'  pairs: {total}  false positives: {fp}  FPR: {fpr:.2f}%')
            logf.write(f'  pairs: {total}\n')
            logf.write(f'  false positives: {fp}\n')
            logf.write(f'  FPR: {fpr:.2f}%\n')
            if failures:
                logf.write(f'  failure list:\n')
                for f in failures:
                    logf.write(f'    - {f}\n')
            logf.write('\n')

        overall_fpr = (fp_all / total_all * 100) if total_all > 0 else 0.0
        print('\n' + '=' * 80)
        print(f'OVERALL: total={total_all}  false_positives={fp_all}  FPR={overall_fpr:.4f}%')
        print('=' * 80)

        logf.write('=' * 80 + '\n')
        logf.write('SUMMARY\n')
        logf.write('=' * 80 + '\n')
        for sub, total, fp, fpr in summary:
            logf.write(f'  {sub:10s}  total={total:4d}  FP={fp:4d}  FPR={fpr:.2f}%\n')
        logf.write(f'  {"OVERALL":10s}  total={total_all:4d}  FP={fp_all:4d}  FPR={overall_fpr:.4f}%\n')
        logf.write(f'\nFinished: {datetime.now()}\n')

    print(f'\nLog: {LOG_PATH}')


if __name__ == '__main__':
    main()
