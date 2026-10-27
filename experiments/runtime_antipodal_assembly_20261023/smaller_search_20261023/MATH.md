# Mathematics of the saved smaller grasp search

Explained 2026-10-26 from the saved 20261024T030119Z run and its matching source.
This documents the historical algorithm; it does not change its settings or
introduce a new selection method. All lengths below are metres unless labeled.

The limits of 24 contact pairs, six hand angles and 12 shortlisted poses were
chosen experiment settings. They are not mathematical optima. The score weights
and spacing thresholds were also hand chosen. The saved comparison supports
their performance in that scene, not a claim that they are the best settings.

## Contact-pair qualification and the 24-pair limit

The sampler casts 512 surface rays around the aligned CAD and constructs
opposing contact pairs. For points p and q, define contact separation w, midpoint
m, and closing direction u:

\[
w=\|q-p\|,\qquad m=(p+q)/2,\qquad u=(q-p)/w.
\]

Local qualification uses both endpoint surface directions n_p and n_q:

\[
|n_p^T u|\ge\cos15^\circ,\quad
|n_q^T u|\ge\cos15^\circ,\quad
n_p^T n_q\le-\cos30^\circ.
\]

Repeated point pairs are removed at 0.00001 m coordinate rounding. Frame
generation requires 0.002 < w < 0.075 m. These are geometric tests, not a
probability of successful pickup or an assembly-error prediction.

Given P qualified pairs, B=min(24,P) entries are selected using NumPy linspace:

\[
i_j=\left\lfloor j(P-1)/(B-1)\right\rfloor,
\quad j=0,\ldots,B-1.
\]

For B=1 the selected index is zero; for B=0 there is no candidate. This spreads
indices through the existing list. It does not optimize 3D surface coverage or
choose the 24 pairs with the best contact quality.

## Hand orientations and finger depths

Project workspace down g=(0,0,-1) onto the plane perpendicular to u:

\[
b=\frac{g-u(u^Tg)}{\|g-u(u^Tg)\|}.
\]

If the denominator is below 0.05, the implementation instead projects workspace
+X. Zero hand angle means as downward as this perpendicularity permits.

For the hand-chosen angle set

\[
\mathcal A=\{0^\circ,45^\circ,-45^\circ,90^\circ,-90^\circ,180^\circ\},
\quad a=\operatorname{Rot}(u,\phi)b,
\]

Rot(u,phi) means rotation by phi around u. Directions with a_z > 0.1 are
discarded. Thus six angles are tested, but fewer may survive. This is not a
uniform sampling of every possible 3D hand orientation.

For each finger assignment s in {+1,-1}, construct the full orientation:

\[
x=su,\qquad y=a\times x,\qquad R=[x\ y\ a].
\]

Changing s rotates the hand 180 degrees around its approach direction and
exchanges the jaws. It keeps the contact line and approach direction unchanged.

The depth set, in implementation order, is D={4,0,-4,-8,6,8} mm. For each d:

\[
t=m-da,\qquad T=\begin{bmatrix}R&t\\0&1\end{bmatrix}.
\]

This puts the contact midpoint at local tool coordinates (0,0,d). Depth changes
the tool position; the orientation remains R. In the smaller search the +6/+8 mm
variants retain the score of their +4 mm parent. Therefore:

\[
N_{\rm poses}\le B\,|\mathcal A|\,2\,|D|
\le24\cdot6\cdot2\cdot6=1728.
\]

The angle and pair limits bound computation. The historical run did not derive
them from an angular-error tolerance or prove them sufficient for every part pose.

## The geometric ordering score

The candidates are ordered using lower-is-preferred score:

\[
J=\frac{\|m-c\|}{0.05}
  +0.25\,\theta(R,R_0)+0.5(1+a_z),
\]

\[
\theta(R,R_0)=\arccos\!\left(
\operatorname{clip}\left((\operatorname{tr}(RR_0^T)-1)/2,-1,1\right)\right).
\]

c is the supplied CAD/body-frame origin in workspace coordinates, not a computed
centroid or centre of mass. R_0 is the initial gripper orientation. Theta is in
radians. The code evaluates it with SciPy's rotation-vector magnitude.

The terms prefer contacts near c, less rotation from R_0, and downward approach.
For a_z=-1 the third term is zero; for horizontal approach it is 0.5. The 50 mm
distance scale and weights 0.25/0.5 are hand-chosen. There is no finger-depth,
contact-area, learned-success or final-assembly term in this score.

## Exact shortlist rule and why its maximum is 12

The static table test below is applied before forming the shortlist. Candidates
that pass are scanned in score order. A pose i is near representative r when:

\[
\|m_i-m_r\|\le0.005,\quad
|x_i^Tx_r|>\cos15^\circ,\quad
a_i^Ta_r>\cos15^\circ.
\]

The absolute value treats exchanged jaws as the same closing line. The first
pass takes one pose per group. If fewer than 12 were collected, a second pass
allows up to two per group. Within a matching group, a second pose is skipped
as redundant if both |d_i-d_j| < 0.001 m and x_i^T x_j > 0.97 for an existing
group member. This allows a new depth or an exchanged-jaw orientation.

The process stops at 12. It is a greedy rule using each group's first pose as
representative, not global clustering or an optimal selection of twelve grasps.
The second pass can also make the final list differ from a single global sort.

## Mathematics of the old gripper/table check

Let V(o) contain every collidable gripper mesh vertex and every box corner,
expressed in the gripper frame at its candidate opening:

\[
o=\min(0.080,w+0.012).
\]

The check uses a horizontal table plane z=z_T with z_T=0 in this run. For pose
(R,t), the lowest gripper clearance at the final grasp is:

\[
h_0=\min_{v\in V(o)} e_z^T(Rv+t)-z_T,
\qquad e_z=(0,0,1).
\]

The open hand's straight 10 cm approach has fixed orientation and opening:

\[
t(s)=t-sa,\qquad 0\le s\le0.10.
\]

Each point's height changes linearly with s, so the minimum along the whole
segment occurs at an endpoint. Therefore the implemented rule is:

\[
h_{\min}=h_0+\min(0,-0.10a_z),\qquad
\text{keep if }h_{\min}\ge0.0001\ \mathrm m.
\]

Thus it requires at least 0.1 mm above the plane. A downward approach a_z=-1
starts above the final grasp, so its lowest point is at the grasp. The test
uses the complete open hand, not just the two contact points. It treats the
table as an infinite horizontal plane and is not a whole-robot path certificate.
The later robot/table contact check and open-finger/target check separately
reject overlaps greater than 0.05 mm; that is a different threshold and stage.

## Why large-gear Grasp 02 was selected

The large gear produced 409 qualified pairs, expanded 24, formed 1,620 poses,
retained 216 after the table test and kept a shortlist of 12.

| Quantity | Grasp 01 | Grasp 02 |
| --- | ---: | ---: |
| Midpoint distance from c | 0.533484 mm | 5.670758 mm |
| Rotation from R_0 | 7.802730 degrees | 5.739138 degrees |
| Distance score term | 0.010669687 | 0.113415160 |
| Rotation score term | 0.034045834 | 0.025041713 |
| Downward score term | approximately zero | approximately zero |
| Saved J | 0.044715522 | 0.138456875 |
| Old table clearance | 5.196518 mm | 10.385674 mm |
| Later pickup-path check | Open fingers overlap gear while lowering | Passed |

Grasp 01 had the better score and both passed the table test. Grasp 01's open
left/right fingers overlapped the target by 4.737912/4.736859 mm, above the
0.05 mm rejection limit. Grasp 02 passed ten checked positions in each of the
approach, lowering and lift segments. Only then was one physical simulation
pickup executed, using Grasp 02.

For ordered shortlist (g_1,...,g_K) and pickup-path predicate F, execution selects:

\[
k^*=\min\{k:F(g_k)=1\},\qquad g^*=g_{k^*}.
\]

This is the first passing candidate in shortlist order. It is not a VLM choice,
the minimum predicted assembly error, or proof that G02 is the best grasp. Its
recorded lifted-and-held height increase was 180.115747 mm.

## Sources and arithmetic verification

- [Contact sampling, pose construction, score and static table test](../grasp_planning.py)
- [Budgets, six depths and two-pass shortlist](search.py)
- [Order of filtering and first-passing pickup selection](run_comparison.py)
- [Large-gear saved record](../../../outputs/runtime_antipodal_assembly_20261023/smaller_search/20261024T030119Z/small/Gear_Large/result.json)

The score terms were replayed from saved contacts and the archived initial
gripper orientation, retaining the original float32 midpoint calculation.
They reproduce both saved scores. No new simulation or grasp search was run.
