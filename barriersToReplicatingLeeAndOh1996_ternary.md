Currently I can't exactly replicate the ternary work of Lee and Oh 1996 (e.g., Figure 9). 
![Current Replication Attempt](fig9_replicationAttempt_2026_05_11.png)

It has proven difficult to rectify this discrepancy. One of the main reasons it is so difficult is that there are many parts of the paper/algorithm that I'm unsure of exactly how to implement. For most of these I have several different implementations that I think it could be (but the correct one may not necessarily be among them). It is also possible that there is another issue beyond these that is responsible for part of the discrepancy (e.g., the phase amounts typo in the paper that was causing a much larger discrepancy). Below I will lay out the core discrepancies, clues, and possible issues.



## Discrepancies
### 1. Normalized thickness vs. Time curve is incorrect
- The Cr and Ni controlled curves diverge fairly early ($\approx10^{-1}$ hours) from each other and the Lee and Oh results
- The Cr and Ni controlled curves settle at different final thicknesses
- Sharp drops occur in the curves (most prevalent at early times) that do not occur in the Lee and Oh results



## Clues
### 1. The mass of the uncontrolled elements drifts much more in my simulations than for Lee and Oh
- I see errors of 1-2% in composition of the uncontrolled element. Whereas Lee and Oh see errors on the order of 0.01%
- These differences mean that the final thickness value must be different than that of Lee and Oh since it must satisfy (a different) equilibrium (technically it's possible for the different overall composition to give result in an equilibrium that gives the same amount of the two phases (and thus same final thickness) as a different equilibrium but that's very unlikely)
### 2. Sharp drops in the normalized thickness curves
- 
- These drops align with when $p$ crosses $p^*$



## Possible Causes of Discrepancies
- s_for_interp
- value of $p^*$
    - They supposedly set $p^*$ to a very small value (e.g., 0.001) at the start of the sim
- 

## Ambiguities
- How to integrate composition profile:
    - methods:
        - weighted, ignore, noIgnore
        - something else
    - how does/should this work for getting the _initialInventory
- How is dt set:
    - Currently -------
- How and when to reconstruct $C_{ignored}$
- What should $C_{m}$ and $C_{m+1}$ (in eqns 22 and 23) be when the interface crosses a node
- Whether to recalculate interfacial flux (used to calculate interface displacement) using the diffused comps (as opposed to using the value from the start calc'ed using the undiffused comps).
    - Could it be that the fluxes and and the compositions of eq 22 are supposed to be from different ones (e.g., fluxes from beginning of time-step and comps from after diffused)?
    - References in the paper to the "broken solid line" and the "thick solid line" would seem to suggest that the compositions should be those after diffused



## Misc

I think this may come down to how to integrate the composition profile: both at the start to get _initialInventory and at the subsequent steps


When attempting to push  $p^*$ close to 0 (or 1):
- runs with initialInventoryMode="phase_length_idealized" have the problem that because the _initialInventory value is not equal to the integral at t=0 the corrections (either "lee_oh_corrected" or "my_corrected") will want to eliminate this difference at the first step and therefore will require a fixed initial displacement which is (or may be) incompatible with a  $p^*$ too close to 0 (or 1)
- There's another issue that can arise even with initialInventoryMode="integrated":
    - crossing $p^*$ causes a previously ignored point to be reconstructed thru interpolation. When integrationMode="weighted" or "noIgnore" (i.e., it is NOT "ignore") this newly interpolated point is included in the integration 
    - This means I should check assumptions (used for justifying weighted approach) about the where the discontinuities in the ignore and noIgnore cases occur with respect to $p$ and $p^*$


Implementing the all-solute mass correction from the Lee 1999 paper could be beneficial for the following reasons:
- If adding this could help in fixing the discrepancies in replicating Figure 9 and, perhaps more importantly, helping me understand what is causing them:
    - If it solves all of the discrepancies outright then we can be fairly sure that they were caused by the mass of the uncontrolled-solute drifting (although we still won't know what caused that drift to be worse than the one in the paper)
    - If it solves some of the discrepancies but some significant ones still remain then we know there are remaining issues (which will we can pinpoint more easily now that others are resolved)
    - If it solves none of the discrepancies (which would be quite surprising) then (assuming it did eliminate the drift of the uncontrolled-solute) we would have to conclude that the discrepancies were not caused by the drift of the uncontrolled-solute
- Furthermore, correcting the mass of all solutes is likely just a better approach and one I should implement and be using regardless of whether it solves the discrepancies in Figure 9




<style>
r { color: Red }
o { color: Orange }
y { color: Yellow }
g { color: Green }
</style>