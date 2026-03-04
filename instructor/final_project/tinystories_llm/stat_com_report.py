import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(report_path="detailed_comparison_report.json"):
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found! Please ensure your evaluation script saved the 'detailed' report.")
        return

    print(f"Reading detailed report from {report_path}...")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        model_name = entry['model']
        persona = entry['persona']
        
        if 'scores' in entry:
            s = entry['scores']
            for metric, val in s.items():
                if val > 0:
                    processed_data.append({
                        "Model": model_name,
                        "Persona": persona,
                        "Metric": metric.capitalize(),
                        "Score": val
                    })

    if not processed_data:
        print("No valid scores found. Check if 'scores' dictionary exists in your JSON.")
        return

    df = pd.DataFrame(processed_data)

    print("\n" + "="*55)
    print("           DETAILED PERFORMANCE METRICS")
    print("="*55)
    summary = df.groupby(['Model', 'Metric'])['Score'].mean().unstack()
    print(summary)
    print("="*55 + "\n")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=df, kind="bar",
        x="Metric", y="Score", hue="Model",
        col="Persona", palette="magma", alpha=.8, height=5, aspect=0.8
    )

    g.set_axis_labels("", "DeepSeek Score (0-10)")
    g.set_titles("{col_name} Style")
    g.despine(left=True)
    g.set(ylim=(0, 11))

    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.1f'), 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 8), 
                               textcoords = 'offset points',
                               fontsize=9, fontweight='bold')

    output_img = "multi_dimension_comparison.png"
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Performance Breakdown: SimpleStory vs TinyStory Baseline", fontsize=16)
    
    plt.savefig(output_img, dpi=300)
    print(f"Analysis complete! Chart saved as: {output_img}")
    plt.show()

if __name__ == "__main__":
    run_analysis()