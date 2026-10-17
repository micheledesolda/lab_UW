"""Recorded reference/current-wave comparison used by the adaptive gate."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.signal.windows import tukey


def record_gate(*, time, reference, current, signal_mask, noise_mask, decision,
                output_directory, prefix, write_plot=True, append_csv=True):
    """Persist the actual fixed-reference pair; no shifting or amplitude refit."""
    time=np.asarray(time,dtype=float);a=np.asarray(reference,dtype=float);b=np.asarray(current,dtype=float)
    signal_mask=np.asarray(signal_mask,dtype=bool);noise_mask=np.asarray(noise_mask,dtype=bool)
    if not (a.shape==b.shape==time.shape==signal_mask.shape==noise_mask.shape):
        raise ValueError('Adaptive comparison array shapes do not match')
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    d=dict(decision)
    j=float(d['data_reference_misfit']);jn=float(d['noise_pair_misfit']);threshold=float(d['effective_threshold'])
    d.update(relative_L2_norm=float(np.sqrt(max(j,0))),relative_L2_squared=j,
             noise_relative_L2_norm=float(np.sqrt(max(jn,0))),
             threshold_relative_L2_norm=float(np.sqrt(max(threshold,0))),
             formula='J = ||w*(d_current-d_reference)||^2 / ||w*d_reference||^2; D=sqrt(J)',
             amplitude_refit=False,time_shift=False,
             signal_window_us=[float(time[signal_mask][0]),float(time[signal_mask][-1])],
             noise_window_us=[float(time[noise_mask][0]),float(time[noise_mask][-1])])
    if append_csv:
        columns=['reference_rec_n','current_rec_n','reference_time_s','current_time_s',
                 'data_reference_misfit','relative_L2_norm','noise_pair_misfit',
                 'effective_threshold','gate_status','gate_confirmation_count']
        path=output/f'{prefix}_gate_decisions.csv';exists=path.exists() and path.stat().st_size>0
        with path.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
            if not exists:w.writeheader()
            w.writerow(d)
    if not write_plot:return d
    stem=f'{prefix}_ref_{int(d["reference_rec_n"])}_rec_n_{int(d["current_rec_n"])}_data_change'
    (output/f'{stem}.json').write_text(json.dumps(d,indent=2)+'\n')
    import matplotlib.pyplot as plt
    ts=time[signal_mask];tn=time[noise_mask]
    fig,axes=plt.subplots(3,1,figsize=(10,8.5),gridspec_kw={'height_ratios':[2,1.5,1]},constrained_layout=True)
    axes[0].plot(ts,a[signal_mask],label=f'Riferimento: rec {d["reference_rec_n"]}',lw=1.6)
    axes[0].plot(ts,b[signal_mask],label=f'Nuova onda: rec {d["current_rec_n"]}',lw=1.4)
    axes[0].set_ylabel('Ampiezza del dato');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    description=d.get('data_description','Confronto che decide l’aggiornamento')
    axes[0].set_title(f'{description} — {d.get("gate_status", "")}\n'
        f'D = {d["relative_L2_norm"]:.5f}; J = {j:.6g}; soglia J = {threshold:.6g}',fontsize=12)
    delta=b[signal_mask]-a[signal_mask];noise_delta=b[noise_mask]-a[noise_mask]
    rms=float(np.sqrt(np.mean(noise_delta**2)))
    axes[1].plot(ts,delta,label='Differenza nel diretto',lw=1.3)
    axes[1].axhspan(-rms,rms,color='gray',alpha=.2,label='± RMS della differenza di rumore')
    axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel('Nuova − riferimento')
    axes[1].set_xlabel('Tempo nella forma d’onda (µs)');axes[1].grid(alpha=.2);axes[1].legend(fontsize=9)
    # The RMS shading is descriptive; it is not a statistical confidence band.
    axes[2].bar(['Differenza nel diretto','Rumore equivalente','Soglia operativa'],[j,jn,threshold],
                color=['#2476a2','#dd8732','#677b68'])
    axes[2].set_ylabel('Norma L2 relativa²');axes[2].grid(axis='y',alpha=.2)
    axes[2].set_ylim(0,1.25*max(j,jn,threshold,1e-15))
    for i,value in enumerate([j,jn,threshold]):axes[2].text(i,value,f'{value:.5g}',ha='center',va='bottom',fontsize=9)
    fig.savefig(output/f'{stem}.png',dpi=160);plt.close(fig)
    return d


def confirm_gate(state, decision, noise_compatible, settings):
    """A moderate exceedance must persist; a strong change updates immediately."""
    if noise_compatible:
        count=0;reuse=True;status='noise_compatible'
    else:
        count=int(state.get('gate_confirmation_count',0))+1
        strong=decision['data_reference_misfit']>=settings['gate_strong_factor']*decision['effective_threshold']
        confirmed=count>=int(settings['gate_confirm_records'])
        reuse=not (strong or confirmed)
        status='strong_change' if strong else ('confirmed_change' if confirmed else 'awaiting_confirmation')
    state['gate_confirmation_count']=count
    decision.update(gate_status=status,gate_confirmation_count=count,
                    gate_confirm_records=int(settings['gate_confirm_records']),
                    gate_strong_factor=float(settings['gate_strong_factor']),
                    raw_noise_compatible=bool(noise_compatible),will_optimize=not reuse)
    return reuse


def reusable_step_limits(limits, reduce_factor):
    """Carry learned step sizes; lift exhausted active steps just above threshold."""
    if not limits:return None
    out={}
    for step,threshold in [('dc_max_start','dc_threshold'),('da_max_start','da_threshold')]:
        value=limits.get(step);floor=limits.get(threshold)
        if value is None or floor is None:return None
        value=float(value);floor=float(floor)
        if not np.isfinite(value+floor) or value<0 or floor<0:return None
        out[step]=max(value,floor*max(float(reduce_factor),1.01)) if value>0 else 0.
        out[threshold]=floor
    return out
