"""Rebuild the paper geometry figure from the recorded canonical diagnostics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def build(output):
    HERE=Path(output);HERE.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('recorded')/'spectral_competitive_california/full_refit/refit_seed0.json'
    b=json.loads(source.read_text())['diagnostics']['banks'][0]
    rho=np.asarray(b['importance']);amp=np.asarray(b['amplitudes']);freq=np.asarray(b['frequencies'])
    assert np.isclose(rho.sum(),1) and np.allclose((amp**2).sum(1),1)
    names=['Median income','House age','Average rooms','Average bedrooms','Population','Average occupancy','Latitude','Longitude']
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.8),sharey=True,gridspec_kw={'width_ratios':[1,1.5]})
    y=np.arange(8)
    axes[0].barh(y,100*rho,color='#356d91');axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlabel('Metric trace mass (%)');axes[0].set_xlim(0,50)
    for j in range(8):
     axes[1].scatter(freq[j],np.full(4,j),s=380*amp[j]**2,color='#b25832',alpha=.85,edgecolors='white',linewidths=.5)
    axes[1].set_xscale('log');axes[1].set_xticks([.02,.05,.1,.2,.5],labels=['0.02','0.05','0.1','0.2','0.5']);axes[1].set_xlabel('Frequency (cycles per standardized unit)')
    for ax in axes:
     ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(HERE/'identifiable_spectral_geometry.pdf',bbox_inches='tight');fig.savefig(HERE/'identifiable_spectral_geometry.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (HERE/'identifiable_spectral_geometry_data.json').write_text(json.dumps(dict(source='recorded/spectral_competitive_california/full_refit/refit_seed0.json',seed=0,feature_names=names,feature_mass=rho.tolist(),frequencies=freq.tolist(),relative_pair_mass=(amp**2).tolist()),indent=2)+'\n')
    print('Geometry figure generated; location trace mass =',float(rho[-2:].sum()))
