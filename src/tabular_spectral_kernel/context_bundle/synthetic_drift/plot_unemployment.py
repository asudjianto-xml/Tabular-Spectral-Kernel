"""Plot the exogenous macro driver, fixed-profile response, and realized error."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import context_runtime as runtime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import unemployment_data as U


def main():
    out=Path(__file__).parent/"unemployment_results"
    d=np.load(out/"seed0/original_localized_downturn.npz")
    months=np.arange(60);rates=d["unemployment_rate"].reshape(60,32)[:,0]
    exposure=d["macro_exposure"].reshape(60,32)[:,0]
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout="constrained")
    axs[0].plot(months,100*rates,color="#555555",lw=2)
    axs[0].set_ylabel("Available unemployment (%)")
    for buffer,label,color in [(.5,"0.5-month buffer","#D55E00"),(3.,"3-month buffer","#0072B2")]:
        raw=np.tile([.8,550,buffer,.5],(60,1))
        x=2*(raw-np.array([0,450,0,0]))/np.array([1,300,4,1])-1
        mu,_=U.conditional_mean(x,exposure,localized=True)
        axs[1].plot(months,100*mu,label=label,color=color,lw=2)
    axs[1].set_ylabel("Expected repayment (%)");axs[1].legend()
    axs[1].set_title("Fixed profiles: 80% utilization, score 550, prior repayment 50%")
    for profile,method,decoder,label in [("original","full128","krr","KRR: full context"),
        ("original","raw_nearest64","krr","KRR: nearest 64"),
        ("nw_calibrated","raw_nearest64","nw","Calibrated NW: nearest 64")]:
        errors=[]
        for seed in [0,1,2]:
            z=np.load(out/f"seed{seed}/{profile}_localized_downturn.npz")
            errors.append(((z[method+"_"+decoder][128:]-z["y"][128:])**2).reshape(56,32).mean(1))
        axs[2].plot(np.arange(4,60),100*np.sqrt(np.mean(errors,axis=0)),label=label)
    axs[2].set_ylabel("Monthly RMSE (pp)");axs[2].set_xlabel("Evaluation month");axs[2].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle("Synthetic credit-card repayment driven by unemployment")
    fig.savefig(out/"unemployment_mechanism.png",dpi=170)
    fig.savefig(out/"unemployment_mechanism.pdf");plt.close(fig)


if __name__=="__main__":main()
