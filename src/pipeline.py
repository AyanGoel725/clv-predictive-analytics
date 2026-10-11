"""Reproducible module entry point: python -m src.pipeline run|train|evaluate|score."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from src.config import ROOT, DATA_PATH, OUTPUTS
from src.data_cleaning import validate_raw_frame, TARGET
from src.model_training import train_and_evaluate_models, latest_run, load_bundle
from src.backtesting import read_evidence, run_backtesting_suite
from src.scoring_pipeline import generate_production_scores
from src.explainability import generate_shap_analysis
from src.submission_analytics import generate_submission_analytics


def evaluate(run_dir=None):
    run_dir = Path(run_dir) if run_dir else latest_run()
    evidence, manifest = read_evidence(run_dir)
    result = run_backtesting_suite(evidence, manifest, run_dir)
    shutil.copyfile(run_dir / 'backtesting_report.md', OUTPUTS / 'backtesting_report.md')
    return result


def score(data=DATA_PATH, run_dir=None, output=None, cash_flows=None, product_mapping=None):
    df, metadata, path = generate_production_scores(data, run_dir=run_dir, output_path=output)
    generate_submission_analytics(path.parent, run_dir, cash_flows, product_mapping)
    # Convenience export is explicitly production scoring, never used for validation.
    if output is None:
        shutil.copyfile(path, OUTPUTS / 'CLV_SCORED_OUTPUT.csv')
        (OUTPUTS / 'latest_score_run.json').write_text(json.dumps({
            'score_run_id': metadata['score_run_id'], 'directory': path.parent.relative_to(OUTPUTS).as_posix(),
        }, indent=2), encoding='utf-8')
    print(f'Scored {len(df)} arrays: {path}', flush=True)
    return df, metadata, path


def refresh_readme_results(run_dir, result):
    readme = ROOT / 'README.md'
    if not readme.exists():
        return
    text = readme.read_text(encoding='utf-8-sig')
    start, end = '<!-- RESULTS_START -->', '<!-- RESULTS_END -->'
    if start not in text or end not in text:
        return
    metrics = result['metrics'].set_index('model')
    names = ['static_5', 'training_prevalence', 'logistic', 'weighted_ensemble_raw',
             'unweighted_xgb_raw', 'unweighted_xgb_sigmoid', 'unweighted_rf_raw', 'unweighted_rf_sigmoid',
             'unweighted_ensemble_raw', 'unweighted_ensemble_sigmoid']
    table = metrics.loc[names, ['mean_probability', 'bias', 'brier', 'roc_auc', 'average_precision', 'brier_lift_vs_5_pct']]
    content = '\n\nRun: ' + Path(run_dir).name + '\n\n' + table.to_markdown(floatfmt='.6f') + '\n\n'
    text = text.split(start)[0] + start + content + end + text.split(end, 1)[1]
    readme.write_text(text, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'train', 'evaluate', 'score'])
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh score CSV path (score command)')
    parser.add_argument('--cash-flows', type=Path, help='Explicit keyed years 1-15 future cash flows; otherwise unit sensitivity only')
    parser.add_argument('--product-mapping', type=Path, help='Actual account/array/product-family lookup')
    args = parser.parse_args(argv)
    if args.command in ('run', 'train'):
        result = train_and_evaluate_models(args.data)
        print(f"Training run saved: {result['run_dir']}", flush=True)
        if args.command == 'train':
            return
        evaluation = evaluate(result['run_dir'])
        bundle = result['bundle']
        raw = validate_raw_frame(args.data)
        generate_shap_analysis(bundle, raw, OUTPUTS / 'shap_plots')
        score(args.data, result['run_dir'], cash_flows=args.cash_flows, product_mapping=args.product_mapping)
        refresh_readme_results(result['run_dir'], evaluation)
    elif args.command == 'evaluate':
        evaluate(args.run_dir)
    else:
        score(args.data, args.run_dir, args.output, args.cash_flows, args.product_mapping)


if __name__ == '__main__':
    main()

