#!/usr/bin/env python3
"""
Machine-verifiable exact certificate for the two-state example in
"Endogenous Strategic Switching: Finite-Step Orbital Stability and Modal Damping".

All displayed six-decimal manuscript parameters are interpreted literally as
rational numbers.  The script verifies:
  1. the exact period-two orbit and strict global-potential selection;
  2. completeness of the support-prefix tree through horizon N=3 using exact
     rational rectangle bounds, including weak selector inequalities;
  3. the five feasible three-step support-sequence regions;
  4. exact negative semidefiniteness of the contraction, radius-return, and
     terminal-active-phase S-procedure certificates via principal minors.

Requirements: Python 3 and sympy.
"""
from pathlib import Path
import csv
import sympy as sp

HERE = Path(__file__).resolve().parent
R = sp.Rational

A = sp.Matrix([[R(845651,10**6), R(154349,10**6)],
               [R(627415,10**6), R(372585,10**6)]])
g1 = sp.Matrix([R(278777,10**6), R(-494849,10**6)])
g2 = sp.Matrix([R(-356179,10**6), R(-1042043,10**6)])
Gamma = sp.diag(R(124212,10**6), R(1758627,10**6))
opts = {"0":sp.Matrix([0,0]), "1":sp.Matrix([1,0]), "2":sp.Matrix([0,1])}
labels = [i+j for i in "012" for j in "012"]

def mode(label):
    label = label.replace("beta","").replace("S","")
    b1,b2 = opts[label[0]],opts[label[1]]
    B = sp.Matrix.hstack(b1,b2)
    S = B.T*B + Gamma
    v = sp.Matrix([(b1.T*g1)[0], (b2.T*g2)[0]])
    F = (sp.eye(2)-B*S.inv()*B.T)*A
    c = B*S.inv()*v
    return B,S,v,F,c

F22,c22 = mode("22")[3:]
F12,c12 = mode("12")[3:]
x0 = sp.factor((sp.eye(2)-F12*F22).inv()*(F12*c22+c12))
x1 = sp.factor(F22*x0+c22)

# Periodic Lyapunov matrices, exactly.
p00,p01,p11,q00,q01,q11 = sp.symbols("p00 p01 p11 q00 q01 q11")
P0s=sp.Matrix([[p00,p01],[p01,p11]])
P1s=sp.Matrix([[q00,q01],[q01,q11]])
eq=[]
for M in [P0s-F22.T*P1s*F22-sp.eye(2),
          P1s-F12.T*P0s*F12-sp.eye(2)]:
    eq += [M[0,0],M[0,1],M[1,1]]
sol=sp.solve(eq,[p00,p01,p11,q00,q01,q11],dict=True)[0]
P0=sp.factor(P0s.subs(sol)); P1=sp.factor(P1s.subs(sol))
Ps=[P0,P1]; centers=[x0,x1]

e1,e2=sp.symbols("e1 e2")
e=sp.Matrix([e1,e2])

def V(P,c,x):
    y=x-c
    return sp.expand((y.T*P*y)[0])

def Q(label,x):
    B,S,v,_,_=mode(label)
    r=v-B.T*(A*x)
    return sp.expand((r.T*S.inv()*r)[0])

def qcoeff(expr):
    p=sp.Poly(sp.expand(expr),e1,e2)
    return tuple(sp.Rational(p.coeff_monomial(m))
                 for m in [e1**2,e1*e2,e2**2,e1,e2,1])

def qeval(cf,x,y):
    a,b,c,d,ee,f=cf
    return a*x*x+b*x*y+c*y*y+d*x+ee*y+f

def qmax_rect(cf,rect):
    xl,xu,yl,yu=rect
    a,b,c,d,ee,f=cf
    pts=[(xl,yl),(xl,yu),(xu,yl),(xu,yu)]
    if c:
        for x in (xl,xu):
            y=-(b*x+ee)/(2*c)
            if yl<=y<=yu: pts.append((x,y))
    if a:
        for y in (yl,yu):
            x=-(b*y+d)/(2*a)
            if xl<=x<=xu: pts.append((x,y))
    det=4*a*c-b*b
    if det:
        x=(b*ee-2*c*d)/det
        y=(b*d-2*a*ee)/det
        if xl<=x<=xu and yl<=y<=yu: pts.append((x,y))
    return max(qeval(cf,x,y) for x,y in pts)

def split(rect):
    xl,xu,yl,yu=rect
    if xu-xl >= yu-yl:
        m=(xl+xu)/2
        return [(xl,m,yl,yu),(m,xu,yl,yu)]
    m=(yl+yu)/2
    return [(xl,xu,yl,m),(xl,xu,m,yu)]

def branch_guards(phase,word):
    x=centers[phase]+e
    guards=[sp.expand(R(1,25)-(e.T*e)[0])]
    guards.append(sp.expand(V(Ps[1-phase],centers[1-phase],x)
                            -V(Ps[phase],centers[phase],x)))
    y=x
    for lab in word:
        qs=Q(lab,y)
        for comp in labels:
            if comp != lab:
                guards.append(sp.expand(qs-Q(comp,y)))
        F,c=mode(lab)[3:]
        y=sp.factor(F*y+c)
    return guards,y

# Known strict rational witnesses for exactly the five nonempty length-3 cells.
known = {
(0,("12","22","12")):(R(-77021,5000000),R(-120667,10000000)),
(0,("22","12","22")):(R(0),R(0)),
(0,("22","22","12")):(R(3,20),R(-1,10)),
(1,("12","12","22")):(R(-1,10),R(0)),
(1,("12","22","12")):(R(0),R(0)),
}
prefix_witness={}
for (ph,w),pt in known.items():
    for L in range(1,4):
        prefix_witness.setdefault((ph,w[:L]),pt)

def exact_feasible(phase,word,max_depth=50,max_boxes=200000):
    guards,_=branch_guards(phase,word)
    coeffs=[qcoeff(g) for g in guards]
    key=(phase,tuple(word))
    if key in prefix_witness:
        pt=prefix_witness[key]
        assert all(qeval(c,*pt)>=0 for c in coeffs)
        return True,0,pt
    stack=[((R(-1,5),R(1,5),R(-1,5),R(1,5)),0)]
    boxes=0
    while stack:
        rect,dep=stack.pop()
        boxes += 1
        if any(qmax_rect(c,rect)<0 for c in coeffs):
            continue
        cx=(rect[0]+rect[1])/2; cy=(rect[2]+rect[3])/2
        if all(qeval(c,cx,cy)>=0 for c in coeffs):
            return True,boxes,(cx,cy)
        if dep>=max_depth or boxes>=max_boxes:
            raise RuntimeError(f"unresolved cell phase={phase}, word={word}")
        stack.extend((r,dep+1) for r in split(rect))
    return False,boxes,None

# Strictness at the exact cycle points.
strict_rows=[]
for phase,x,sel in [(0,x0,"22"),(1,x1,"12")]:
    qsel=Q(sel,x)
    diffs=[(sp.factor(qsel-Q(c,x)),c) for c in labels if c!=sel]
    margin,competitor=min(diffs,key=lambda z: float(z[0]))
    assert margin>0
    strict_rows.append((phase,"beta"+sel,"beta"+competitor,margin))
print("STRICTNESS: PASS")
for ph,sel,comp,m in strict_rows:
    print(f"  phase {ph}: {sel} beats {comp}; margin = {sp.N(m,12)}")

# Complete exact prefix tree through N=3.
feasible_prev={(0,()),(1,())}
tree=[]
for L in (1,2,3):
    feasible_now=set()
    for phase,prefix in sorted(feasible_prev):
        for lab in labels:
            word=prefix+(lab,)
            ok,boxes,pt=exact_feasible(phase,word)
            tree.append((L,phase,word,ok,boxes,pt))
            if ok: feasible_now.add((phase,word))
    feasible_prev=feasible_now

finals=sorted((ph,w) for ph,w in feasible_prev)
expected=sorted([
(0,("12","22","12")),
(0,("22","12","22")),
(0,("22","22","12")),
(1,("12","12","22")),
(1,("12","22","12")),
])
assert finals==expected
print("PREFIX TREE / COMPLETENESS: PASS")
print("  exactly five feasible length-3 cells")

with open(HERE/"verified_branch_tree.csv","w",newline="") as f:
    wr=csv.writer(f)
    wr.writerow(["length","phase","sequence","status","boxes_checked","witness_e1","witness_e2"])
    for L,ph,w,ok,boxes,pt in tree:
        wr.writerow([L,ph,"-".join("beta"+z for z in w),
                     "FEASIBLE" if ok else "INFEASIBLE",boxes,
                     "" if pt is None else str(pt[0]),
                     "" if pt is None else str(pt[1])])

def qmatrix(expr):
    p=sp.Poly(sp.expand(expr),e1,e2)
    M=sp.zeros(3)
    M[0,0]=p.coeff_monomial(e1**2)
    M[1,1]=p.coeff_monomial(e2**2)
    M[0,1]=M[1,0]=p.coeff_monomial(e1*e2)/2
    M[0,2]=M[2,0]=p.coeff_monomial(e1)/2
    M[1,2]=M[2,1]=p.coeff_monomial(e2)/2
    M[2,2]=p.coeff_monomial(1)
    return M

def target(phase,word,terminal,kind):
    guards,y=branch_guards(phase,word)
    if kind=="contraction":
        return sp.expand(V(Ps[terminal],centers[terminal],y)
                         -R(31,50)*V(Ps[phase],centers[phase],centers[phase]+e))
    if kind=="radius":
        z=y-centers[terminal]
        return sp.expand((z.T*z)[0]-R(1,25))
    return sp.expand(V(Ps[terminal],centers[terminal],y)
                     -V(Ps[1-terminal],centers[1-terminal],y))

def parse_rat(s):
    return sp.Rational(s)

mult=list(csv.DictReader(open(HERE/"exact_multipliers.csv")))
terminal={
(0,("12","22","12")):0,
(0,("22","12","22")):1,
(0,("22","22","12")):0,
(1,("12","12","22")):1,
(1,("12","22","12")):0,
}
cert_map=[("contraction","contraction_eta_31_50"),
          ("radius","return_to_radius_1_5"),
          ("phase","terminal_phase_active")]

lmi_rows=[]
for (phase,word),term in terminal.items():
    guards,_=branch_guards(phase,word)
    seq="-".join("beta"+z for z in word)
    for kind,cname in cert_map:
        M=qmatrix(target(phase,word,term,kind))
        for row in mult:
            if (row["certificate"]==cname and int(row["phase"])==phase
                and row["sequence"]==seq and int(row["terminal_phase"])==term):
                M += parse_rat(row["tau"])*qmatrix(guards[int(row["guard_index"])])
        N=-sp.simplify(M)
        idx=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]
        minors=[sp.factor(N.extract(I,I).det()) for I in idx]
        assert all(v>=0 for v in minors)
        lmi_rows.append((phase,seq,kind,term,minors))
print("S-PROCEDURE LMIs: PASS")
print(f"  {len(lmi_rows)} exact 3x3 semidefinite implications verified")

with open(HERE/"verified_lmi_principal_minors.csv","w",newline="") as f:
    wr=csv.writer(f)
    wr.writerow(["phase","sequence","certificate","terminal_phase",
                 "m0","m1","m2","m01","m02","m12","m012"])
    for ph,seq,kind,term,mins in lmi_rows:
        wr.writerow([ph,seq,kind,term]+[str(v) for v in mins])

print("ALL EXACT CHECKS PASSED")
