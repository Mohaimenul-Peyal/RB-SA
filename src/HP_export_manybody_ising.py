#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HP_export_manybody_ising.py

Exports HP many-body model in both QUBO (x in {0,1}) and k-local Ising (s in {-1,1})
without reducing higher-order terms. Outputs JSON with linear/quadratic/cubic/quartic
QUBO terms plus Ising terms (h, J2, J3, J4).
"""
import json, math, argparse
from pathlib import Path
from itertools import combinations
import numpy as np

def get_args():
    p = argparse.ArgumentParser(description="Export HP many-body QUBO + k-local Ising.")
    p.add_argument("--seq", required=True)
    p.add_argument("--l_size", type=int, default=None)
    p.add_argument("--auto_mode", type=str, default="balanced", choices=["tight","balanced","roomy"])
    p.add_argument("--hp_reward", type=float, default=-1.0)
    p.add_argument("--enable_3body", action="store_true")
    p.add_argument("--enable_4body", action="store_true")
    p.add_argument("--gamma_3", type=float, default=-0.5)
    p.add_argument("--gamma_4", type=float, default=-0.3)
    p.add_argument("--penalty_mode", type=str, default="auto", choices=["auto","manual"])
    p.add_argument("--lambda_onehot", type=float, default=None)
    p.add_argument("--lambda_site", type=float, default=None)
    p.add_argument("--lambda_chain", type=float, default=None)
    p.add_argument("--scale_target", type=float, default=1000.0)
    p.add_argument("--out", type=str, default="qubo_manybody_ising.json")
    return p.parse_args()

def choose_L(N, mode="roomy", minL=3):
    factors={"tight":1.0,"balanced":1.5,"roomy":2.0}
    return max(minL, int(math.ceil(math.sqrt(factors.get(mode,2.0)*N))))

def lattice_coords(L): return [(x,y) for x in range(L) for y in range(L)]

def neighbor_map(coords):
    sset=set(coords); n={c:set() for c in coords}
    for x,y in coords:
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb=(x+dx,y+dy)
            if nb in sset: n[(x,y)].add(nb)
    return n

def plaquettes(L):
    out=[]
    for x in range(L-1):
        for y in range(L-1):
            out.append(((x,y),(x+1,y),(x,y+1),(x+1,y+1)))
    return out

def auto_penalties(seq,hp_reward,g3,e3,g4,e4):
    H=seq.count("H"); deg=4
    C_local=deg*max(H-1,0)
    T_local=6*max(H-2,0) if e3 else 0
    P_local=8*max(H-3,0) if e4 else 0
    margin=3.0
    base_mag=abs(hp_reward)*C_local+abs(g3)*T_local+abs(g4)*P_local
    lam_anc=max(1.0,2.0*base_mag*margin)
    return 1.5*lam_anc, 1.5*lam_anc, 1.5*lam_anc

def add_ising_from_qubo(idx_list, coeff, h, J2, J3, J4):
    """Map c * prod x_i to Ising s via x=(s+1)/2; k<=4."""
    k=len(idx_list)
    if k==0: return
    factor = coeff / (2**k)
    from itertools import combinations
    for t in range(1,k+1):
        for subset in combinations(idx_list,t):
            if t==1:
                h[subset[0]] = h.get(subset[0],0.0) + factor
            elif t==2:
                a,b=sorted(subset)
                J2[(a,b)] = J2.get((a,b),0.0) + factor
            elif t==3:
                key=tuple(sorted(subset))
                J3[key] = J3.get(key,0.0) + factor
            elif t==4:
                key=tuple(sorted(subset))
                J4[key] = J4.get(key,0.0) + factor

def main():
    args=get_args()
    SEQ=args.seq.strip().upper(); N=len(SEQ)
    L=int(args.l_size) if args.l_size is not None else choose_L(N, args.auto_mode)
    coords=lattice_coords(L); S=len(coords); site_to_idx={s:i for i,s in enumerate(coords)}
    nbrs=neighbor_map(coords); cells=plaquettes(L)
    if args.penalty_mode=="manual":
        if None in (args.lambda_onehot,args.lambda_site,args.lambda_chain):
            raise SystemExit("manual penalties require all lambda_*")
        lam1h=args.lambda_onehot; lams=args.lambda_site; lamc=args.lambda_chain
    else:
        lam1h,lams,lamc=auto_penalties(SEQ,args.hp_reward,args.gamma_3,args.enable_3body,args.gamma_4,args.enable_4body)

    linear=[]; quad=[]; cubic=[]; quartic=[]
    def idx_of(r,site): return r*S+site_to_idx[site]

    for r in range(N):
        inds=[idx_of(r,s) for s in coords]
        for i in inds: linear.append((i,-lam1h))
        for a in range(len(inds)):
            i=inds[a]
            for b in range(a+1,len(inds)):
                j=inds[b]; quad.append((min(i,j),max(i,j),2.0*lam1h))
    for s in coords:
        inds=[idx_of(r,s) for r in range(N)]
        for a in range(len(inds)):
            i=inds[a]
            for b in range(a+1,len(inds)):
                j=inds[b]; quad.append((min(i,j),max(i,j),lams))
    for r in range(N-1):
        for s1 in coords:
            i1=idx_of(r,s1)
            nbr_next={idx_of(r+1,s2) for s2 in nbrs[s1]}
            all_next={idx_of(r+1,s2) for s2 in coords}
            for i2 in (all_next - nbr_next):
                quad.append((min(i1,i2),max(i1,i2),lamc))
    for i_res in range(N):
        if SEQ[i_res]!="H": continue
        for j_res in range(i_res+2,N):
            if SEQ[j_res]!="H": continue
            for s1 in coords:
                i1=idx_of(i_res,s1)
                for s2 in nbrs[s1]:
                    i2=idx_of(j_res,s2)
                    quad.append((min(i1,i2),max(i1,i2),args.hp_reward))

    if args.enable_3body:
        triples=[]
        for s1 in coords:
            nbs=list(nbrs[s1])
            for a in range(len(nbs)):
                for b in range(a+1,len(nbs)):
                    s2,s3=nbs[a],nbs[b]
                    v1=(s2[0]-s1[0],s2[1]-s1[1]); v2=(s3[0]-s1[0],s3[1]-s1[1])
                    if v1[0]*v2[0]+v1[1]*v2[1]!=0: continue
                    triples.append((s1,s2,s3))
        for i in range(N):
            if SEQ[i]!="H": continue
            for j in range(i+1,N):
                if SEQ[j]!="H" or abs(j-i)<=1: continue
                for k in range(j+1,N):
                    if SEQ[k]!="H" or abs(k-j)<=1 or abs(k-i)<=1: continue
                    for (s1,s2,s3) in triples:
                        xi=idx_of(i,s1); xj=idx_of(j,s2); xk=idx_of(k,s3)
                        cubic.append(tuple(sorted((xi,xj,xk))) + (args.gamma_3,))

    if args.enable_4body:
        Hidx=[i for i,t in enumerate(SEQ) if t=="H"]
        quads=[q for q in combinations(Hidx,4) if all(abs(q[u]-q[v])>1 for u in range(4) for v in range(u+1,4))]
        for cell in cells:
            sA,sB,sC,sD=cell
            for (i,j,k,l) in quads:
                xi=idx_of(i,sA); xj=idx_of(j,sB); xk=idx_of(k,sC); xl=idx_of(l,sD)
                quartic.append(tuple(sorted((xi,xj,xk,xl))) + (args.gamma_4,))

    def max_abs():
        vals=[abs(c) for _,c in linear]+[abs(c) for *_,c in quad]+[abs(c) for *_,c in cubic]+[abs(c) for *_,c in quartic]
        return max(vals) if vals else 0.0
    alpha = 1.0
    maxc=max_abs()
    if maxc>0: alpha=float(args.scale_target)/maxc
    if alpha!=1.0:
        linear=[(i,c*alpha) for i,c in linear]
        quad=[(i,j,c*alpha) for i,j,c in quad]
        cubic=[(a,b,cidx,c*alpha) for a,b,cidx,c in cubic]
        quartic=[(a,b,cidx,d,c*alpha) for a,b,cidx,d,c in quartic]

    h={}; J2={}; J3={}; J4={}
    for i,c in linear: add_ising_from_qubo([i], c, h,J2,J3,J4)
    for i,j,c in quad: add_ising_from_qubo([i,j], c, h,J2,J3,J4)
    for a,b,cidx,c in cubic: add_ising_from_qubo([a,b,cidx], c, h,J2,J3,J4)
    for a,b,cidx,d,c in quartic: add_ising_from_qubo([a,b,cidx,d], c, h,J2,J3,J4)

    out=dict(
        format="manybody_qubo_ising_v1",
        seq=SEQ, N=N, L=L, S=S, V=N*S,
        enable_3body=bool(args.enable_3body),
        enable_4body=bool(args.enable_4body),
        hp_reward=float(args.hp_reward),
        gamma_3=float(args.gamma_3),
        gamma_4=float(args.gamma_4),
        lambda_onehot=float(lam1h),
        lambda_site=float(lams),
        lambda_chain=float(lamc),
        scale_target=float(args.scale_target),
        scale_alpha=float(alpha),
        coords=coords,
        linear=linear, quadratic=quad, cubic=cubic, quartic=quartic,
        ising=dict(
            h=h,
            J2=[(i,j,c) for (i,j),c in J2.items()],
            J3=[(*k,c) for k,c in J3.items()],
            J4=[(*k,c) for k,c in J4.items()]
        )
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {args.out} (QUBO + k-local Ising). alpha={alpha:.4g}")

if __name__=="__main__":
    main()

