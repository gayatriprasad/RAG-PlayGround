"""
Experiment reporter for saving results and generating summaries.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ExperimentReporter:
    """
    Reports experiment results as CSV, printed summaries, and markdown reports.
    """
    
    def save_csv(self, df, out_dir: str, experiment_name: str) -> str:
        """
        Save results DataFrame to CSV.
        
        Args:
            df: pandas DataFrame with scored results
            out_dir: Output directory path
            experiment_name: Name of the experiment
            
        Returns:
            Path to saved CSV file
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        csv_path = out_path / f"{experiment_name}_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to: {csv_path}")
        return str(csv_path)
    
    def print_summary(self, df) -> None:
        """
        Print pivot table summaries of results.
        
        Prints:
        - source_type × pipeline: mean overall_score
        - category × pipeline: mean overall_score
        - Highlights top performer per row
        
        Args:
            df: pandas DataFrame with scored results
        """
        import pandas as pd
        
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 70)
        
        # Overall stats
        total = len(df)
        correct = df['answer_correct'].sum() if 'answer_correct' in df.columns else 0
        mean_score = df['overall_score'].mean() if 'overall_score' in df.columns else 0
        
        accuracy = (correct / total * 100) if total > 0 else 0.0

        print(f"\nTotal questions: {total}")
        print(f"Correct answers: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Mean overall score: {mean_score:.3f}")

        if total == 0:
            print("\nNo results to summarize (all questions may have been skipped by resume).")
            print("\n" + "=" * 70)
            return
        
        # Pivot: source_type × pipeline
        if 'source_type' in df.columns and 'pipeline' in df.columns:
            print("\n─── By Source Type × Pipeline ───")
            try:
                pivot_source = pd.pivot_table(
                    df,
                    values='overall_score',
                    index='source_type',
                    columns='pipeline',
                    aggfunc='mean'
                )
                print(pivot_source.to_string(float_format="{:.3f}".format))
                
                # Highlight top performer per source_type
                if len(pivot_source.columns) > 1:
                    print("\n  Top performer per source_type:")
                    for idx in pivot_source.index:
                        row = pivot_source.loc[idx]
                        best = row.idxmax()
                        print(f"    {idx}: {best} ({row[best]:.3f})")
            except Exception as e:
                logger.debug(f"Could not create source pivot: {e}")
        
        # Pivot: category × pipeline
        if 'category' in df.columns and 'pipeline' in df.columns:
            print("\n─── By Category × Pipeline ───")
            try:
                pivot_cat = pd.pivot_table(
                    df,
                    values='overall_score',
                    index='category',
                    columns='pipeline',
                    aggfunc='mean'
                )
                print(pivot_cat.to_string(float_format="{:.3f}".format))
                
                # Highlight top performer per category
                if len(pivot_cat.columns) > 1:
                    print("\n  Top performer per category:")
                    for idx in pivot_cat.index:
                        row = pivot_cat.loc[idx]
                        best = row.idxmax()
                        print(f"    {idx}: {best} ({row[best]:.3f})")
            except Exception as e:
                logger.debug(f"Could not create category pivot: {e}")
        
        print("\n" + "=" * 70)
    
    def save_markdown_report(
        self,
        df,
        out_dir: str,
        experiment_name: str,
        config_snapshot: Optional[dict] = None
    ) -> str:
        """
        Save a markdown report with tables and analysis.
        
        Args:
            df: pandas DataFrame with scored results
            out_dir: Output directory path
            experiment_name: Name of the experiment
            config_snapshot: Optional dict of config for inclusion
            
        Returns:
            Path to saved markdown file
        """
        import pandas as pd
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        md_path = out_path / f"{experiment_name}_report.md"
        
        lines = []
        lines.append(f"# Experiment Report: {experiment_name}")
        lines.append("")
        
        # Summary stats
        total = len(df)
        correct = df['answer_correct'].sum() if 'answer_correct' in df.columns else 0
        mean_score = df['overall_score'].mean() if 'overall_score' in df.columns else 0
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total questions | {total} |")
        accuracy = (correct / total * 100) if total > 0 else 0.0
        lines.append(f"| Correct answers | {correct}/{total} ({accuracy:.1f}%) |")
        lines.append(f"| Mean overall score | {mean_score:.3f} |")
        lines.append("")
        
        # Pivot tables
        if 'source_type' in df.columns and 'pipeline' in df.columns:
            lines.append("## Results by Source Type")
            lines.append("")
            try:
                pivot = pd.pivot_table(
                    df, values='overall_score',
                    index='source_type', columns='pipeline',
                    aggfunc='mean'
                )
                lines.append(pivot.to_markdown(floatfmt=".3f"))
            except Exception:
                lines.append("*Could not generate pivot table*")
            lines.append("")
        
        if 'category' in df.columns and 'pipeline' in df.columns:
            lines.append("## Results by Category")
            lines.append("")
            try:
                pivot = pd.pivot_table(
                    df, values='overall_score',
                    index='category', columns='pipeline',
                    aggfunc='mean'
                )
                lines.append(pivot.to_markdown(floatfmt=".3f"))
            except Exception:
                lines.append("*Could not generate pivot table*")
            lines.append("")
        
        # Top 5 questions
        lines.append("## Top 5 Questions (Highest Score)")
        lines.append("")
        if len(df) > 0 and 'overall_score' in df.columns:
            top5 = df.nlargest(5, 'overall_score')[['question_id', 'overall_score', 'pipeline', 'category']]
            lines.append(top5.to_markdown(index=False, floatfmt=".3f"))
        else:
            lines.append("*No rows available*")
        lines.append("")
        
        # Bottom 5 questions
        lines.append("## Bottom 5 Questions (Lowest Score)")
        lines.append("")
        if len(df) > 0 and 'overall_score' in df.columns:
            bottom5 = df.nsmallest(5, 'overall_score')[['question_id', 'overall_score', 'pipeline', 'category']]
            lines.append(bottom5.to_markdown(index=False, floatfmt=".3f"))
        else:
            lines.append("*No rows available*")
        lines.append("")
        
        # Config snapshot
        if config_snapshot:
            lines.append("## Configuration")
            lines.append("")
            lines.append("```yaml")
            import yaml
            lines.append(yaml.dump(config_snapshot, default_flow_style=False))
            lines.append("```")
            lines.append("")
        
        # Write file
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report saved to: {md_path}")
        return str(md_path)
