"""Numbers behind the midterm deck, from the files in results/.

Usage: python -m src.summarize_runs results

Merges the per-batch sweep files, then prints throughput, the arm comparison,
agent step budgets, threshold sweeps from saved scores, NFCorpus and Qwen 1.5B.
"""
import json,glob,sys
import pandas as pd, numpy as np
pd.set_option('display.width',190,'display.max_columns',40,'display.max_rows',80)
R=sys.argv[1]
def L(f): return pd.DataFrame(json.loads(l) for l in open(f) if l.strip())
sw=pd.concat([L(f) for f in sorted(glob.glob(R+'/20newsgroups_sweep_bs*.jsonl')) if 'scores' not in f])
print('gpus',sw.gpu.unique(),'batches',sorted(sw.batch_size.unique()))
c=sw[sw.arm=='classify']
print('\n== classify docs/s by N x batch (mean of per-query)')
print(c.pivot_table(index='batch_size',columns='corpus_size',values='n_docs_per_sec').round(1))
print('host per batch', sw.groupby('batch_size').host.first().str.replace('.chtc.wisc.edu','').to_dict())
print('tokens/s', c.groupby('batch_size').n_prefill_tokens_per_sec.mean().round(0).to_dict())
b=sw[sw.batch_size==32]
print('\n== arms bs=32')
print(b.groupby(['corpus_size','arm'])[['recall','precision','f1','best_f1','recall@100','recall@1000','wall_s','n_docs_scored','ann_recall_ceiling']].mean().round(3))
print('\n== agent by budget')
for n,f in [(8,'agent'),(32,'agent_s32'),(64,'agent_s64')]:
    a=L(f'{R}/20newsgroups_{f}.jsonl')
    print(n,a.groupby('corpus_size').agg(rec=('recall','mean'),prec=('precision','mean'),s_q=('wall_s','mean'),steps=('n_steps','mean'),calls=('n_llm_calls','mean'),docs=('n_docs_scored','mean'),found=('tp','mean'),gold=('n_gold','mean'),chose=('stopped_early','mean')).round(3).to_string())
a=L(f'{R}/20newsgroups_agent_s64.jsonl')
print('s64 sec/call',round(a.t_llm.sum()/a.n_llm_calls.sum(),3),'gen/call',round(a.n_gen_tokens.sum()/a.n_llm_calls.sum(),1),'prompt/call',round(a.n_prompt_tokens.sum()/a.n_llm_calls.sum()))
a8=L(f'{R}/20newsgroups_agent.jsonl'); print('s8 sec/call',round(a8.t_llm.sum()/a8.n_llm_calls.sum(),3))
print('max steps s64',a.n_steps.max(), a.groupby('corpus_size').n_steps.max().to_dict())
def op(path,label):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    d=pd.DataFrame(r for r in rows if r['arm']=='classify')
    for n,g in d.groupby('corpus_size'):
        out=[]
        for t in [.5,.1,.01,.001,1e-4]:
            P=[];Rc=[];F=[];pred=0
            for _,r in g.iterrows():
                gold=set(r['gold']); p={k for k,v in r['scores'].items() if v>=t}
                tp=len(p&gold); pr=tp/len(p) if p else 0; rc=tp/len(gold)
                P.append(pr);Rc.append(rc);F.append(2*pr*rc/(pr+rc) if pr+rc else 0);pred+=len(p)
            out.append((t,round(np.mean(Rc),3),round(np.mean(P),3),round(np.mean(F),3)))
        print(label,n,'(t,R,P,F1):',out)
        v=np.concatenate([np.array(list(r['scores'].values())) for _,r in g.iterrows()])
        print('   score zero frac',round((v==0).mean(),3),' <1e-3',round((v<1e-3).mean(),3),' >.5',round((v>.5).mean(),3))
op(R+'/20newsgroups_sweep_bs32.scores.jsonl','7B')
op(R+'/20newsgroups_small.scores.jsonl','1.5B')
op(R+'/nfcorpus_nf.scores.jsonl','NF')
nf=L(R+'/nfcorpus_nf.jsonl')
print('\n== NFCorpus'); print(nf.groupby('arm')[['recall','precision','f1','best_f1','recall@10','recall@100','ndcg@10','wall_s','n_docs_scored']].mean().round(3)); print('gold median',nf[nf.arm=='ann'].n_gold.median())
sm=L(R+'/20newsgroups_small.jsonl')
print('\n== 1.5B'); print(sm.groupby(['corpus_size','arm'])[['recall','precision','recall@100','wall_s']].mean().round(3))
print('1.5B sec/call',round(sm[sm.arm=='agent'].t_llm.sum()/sm[sm.arm=='agent'].n_llm_calls.sum(),3), 'parse fails',sm[sm.arm=='agent'].n_parse_failures.sum())
print('gpu mem peak', c.groupby('batch_size').peak_gpu_mb.mean().round(0).to_dict())
