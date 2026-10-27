"""Standalone scientific comparison figure from the completed audit only."""
from common import HERE, read
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    output=HERE/'analysis'
    rows=read(output/'per_object.json')
    order=['Gear_Large','M16_Hex_Nut','Waterproof_Male','USB_Male','DSUB_Male','BNC_Male',
           'RGOCG8-50_8mm','RGOCG12-50_12mm','RGOCG16-50_16mm','KET8_Square_8mm','KET12_Square_12mm','KET16_Square_16mm']
    rows.sort(key=lambda r:order.index(r['target']))
    summary=read(output/'summary.json')
    methods=list(summary)
    label_map={'ours':'Ours\nCLIP + depth','cnos':'CNOS','sam6d_rgb':'SAM-6D\nRGB','sam6d_rgbd':'SAM-6D\nRGB-D',
               'nids_basic':'NIDS-Net\nnative failure','nids_basic_finite_templates':'NIDS-Net\nfinite-ref control','noctis':'NOCTIS','nemo':'NeMO'}
    labels=[label_map[m] for m in methods]
    names={'Gear_Large':'Large gear','M16_Hex_Nut':'M16 hex nut','Waterproof_Male':'Waterproof connector',
           'USB_Male':'USB connector + cable','DSUB_Male':'D-sub connector','BNC_Male':'BNC connector',
           'RGOCG16-50_16mm':'Round pin, 16 mm','RGOCG12-50_12mm':'Round pin, 12 mm','RGOCG8-50_8mm':'Round pin, 8 mm',
           'KET16_Square_16mm':'Rectangular pin, 16 mm','KET12_Square_12mm':'Rectangular pin, 12 mm','KET8_Square_8mm':'Rectangular pin, 8 mm'}
    values=np.array([[np.nan if m=='nids_basic' else r[m] for m in methods] for r in rows])
    fig,axes=plt.subplots(2,1,figsize=(14,12),gridspec_kw={'height_ratios':[1,2.4]})
    ax=axes[0]
    counts=[np.nan if m=='nids_basic' else summary[m]['correct'] for m in methods]
    bars=ax.bar(np.arange(len(methods)),counts,color=['#00738b',*(['#7592b0']*(len(methods)-1))],width=.65)
    ax.set_ylim(0,105);ax.set_ylabel('Correct requests / 96');ax.set_xticks(range(len(methods)),labels)
    ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,counts):
        if np.isfinite(value):ax.text(bar.get_x()+bar.get_width()/2,value+2,str(int(value)),ha='center',weight='bold',fontsize=12)
        else:ax.text(bar.get_x()+bar.get_width()/2,6,'NaN\nreferences',ha='center',color='#7c3030',fontsize=11)
    ax.set_title('Full-scene CAD identification: own masks for every method',loc='left',fontsize=20,pad=20,weight='bold')
    ax=axes[1]
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#e3e3e3')
    ax.imshow(values,vmin=0,vmax=8,cmap=cmap,aspect='auto')
    ax.set_xticks(range(len(methods)),labels,fontsize=11)
    ax.set_yticks(range(len(rows)),[names.get(r['target'],r['target']) for r in rows],fontsize=11)
    ax.tick_params(length=0)
    for i in range(len(rows)):
        for j in range(len(methods)):
            value=values[i,j]
            text='N/A' if not np.isfinite(value) else f'{int(value)}/8'
            ax.text(j,i,text,ha='center',va='center',color='white' if value>=5 else '#172837',fontsize=12,weight='bold')
    for edge in ax.spines.values():edge.set_visible(False)
    fig.text(.02,.015,'Same 8 development simulation scenes and 12 CADs. Correct = requested identity and target box IoU ≥ 0.5.\nNative NIDS has invalid references; control excludes only six nonfinite CAD-view scores. Ours retains unavailable depth decisions.\nFrozen weights; native proposal/filtering settings; custom CAD-render reference banks. This is not BOP AP.',fontsize=10,color='#405266')
    fig.tight_layout(rect=(0,.06,1,1),h_pad=3)
    fig.savefig(output/'comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'comparison.pdf',bbox_inches='tight')


if __name__=='__main__':main()
