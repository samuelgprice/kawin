


## Olaye & Ojo 2020 "Leapfrog/Dufort-Frankel explicit scheme for diffusion-controlled moving interphase boundary problems with variable diffusion coefficient and solute conservation"

### Objectives:
- Implement binary version (which paper outlines)
- Extend to ternary version (which paper does not outline)


### Current plan:
- first sketch out the ternary extension to confirm that it seems feasible (<r>deadline: 5/29</r>)
    - status (<b>6/1</b>): I think extending the Olaye Dufort-Frankel scheme to a ternary is feasible but it does result in some real differences:
        - have to do a 2X2 matrix solve to update the composition of each grid point (so not exactly explicit anymore but still better then fully global implicit since it is done locally)
        - the interface equation becomes "extra" nonlinear meaning that exact roots can't be used (as Olaye does) and there are two unknowns s^(k+1) and the tieline parameter making the solving more tricky (can't use simple 1D root finding methods)
    - So I think it's possible to get working code based on this. However, it's plausible that these differences make the resulting implementation inefficient, perhaps to the point of being impractical. That said I see I think it's still worth pursuing so I will work on implementing the binary version (from the paper) now and then the ternary afterwards.
- then implement binary version:
    - compare results to those in Olaye:
        - (<r>deadline: 6/5</r>) <pu>Replicate data in figure 5 and figure 6</pu>
            - Seems that the choice of dt relative to dx (inverse of number u/v points) sig affects results. If dt is too large relative to dx then wild results occur. 
            Makes sense because of the limits on dt set by the consistency requirements. However, there are still likely many errors in the code still:
                - [x] Using eqn 18 as written in paper but likely should be modified to work with non-uniform timesteps
                  - [ ] verify this actually helps by trying rework but with nonuniform time correction turned off
                - [ ] check the mass conservation by doing control volume sum (likely in "_lfdf_phase_update")
                  - Seems that initial mass will be off because the interface compositions are given a half-cell of space (same issue that I had with Lee and Oh stuff)
                  - Mass also changes during the run. <r> Need to figure out why that is </r>
                    - <g> Winding equation 12 helps mitigate the drift greatly! </g>
                - [ ] Could email authors with questions
                  - what dx (grid spacing), dt, dt0 was used
                  - was winding applied to eqn 12? If not wouldn't that result in mass drift
                  - What does "Fully Implicit" refer to in Figrue 6? How is it different from "Illingworth"
                  - 
                - [ ] Calculate and Inspect CFL constants: mu, w
        - [ ] Compare to Illingworth and figure out why it seems slower than Illingworth when Olaye claims the opposite
        - [x] (<r>deadline: 7/17</r>) Fix overreliance on physical grid
        - [ ] (<r>deadline: 7/17</r>) Compare early-time behavior to analytic solution
        - [ ] (<r>deadline: 7/22</r>) Replicate Figure 6
    - Attempted to replicate figure 6:
      - Early-time behavior (basically up to peak) deviated from extracted curves but this occurred will the Illingworth model (and data from other Illingworth paper Figure 3). However, my calcualtions matched the analytical solution and the C++ code results nearly exactly so I think it's something going on with their plotting/extract or maybe the convergence

    - compare results to those of Lee binary
- then implement ternary version
    - compare results to Lee ternary

## Illingworth & Golosnoy 2005 "Numerical solutions of diffusion-controlled moving boundary problems which conserve solute"

### Current plan:
- Binary Implementation
  - [x] Add non-uniform time-stepping once confirmed that it acceptable to do so
    - This worked well
  - [x] Sped-up binary implementation by ~100x:
    - Rewrote "_new_concentration_left_planar" and "_new_concentration_right_planar" to be vectorized
    - Rewrote "solve_illingworth_tridiagonal" with numba 
  - [ ] Address issue of non-convergence of solve iterations near disappearance of phase
    - Increasing number of nodes in phase that is going to zero causes the issue sooner
  - [ ] (<r>deadline: 7/17</r>) Compare early-time behavior to analytic solution
  - [ ] (<r>deadline: 7/18</r>) Validate against Figures 4, 5, and 6
    - Added irregular grid capability in the process
    - 


- [ ] Ternary Extension
  - [ ]  (<r>deadline: 7/7</r>) <pu> Determine if ternary extension of Illingworth is possible/practical</pu>
    - [x] Consult old chats
    - [ ] Look at code Claude supposedly made for it
    - [ ] Ask sol about it
  - [ ] Actual Attempt:
    - [ ] (<r>deadline: 7/20</r>) <pu> Ask codex to attempt an implementation of a ternary extension **before showing it Ian's Claude attempt**
  - [ ] 

## Lee work: 1996 and 1999 papers:

- <r> Maybe try totally different way of integrating composition profile </r>:
    - instead of using trapezoid maybe just do composition times cell width (akin to finite volume approach)
- [ ] Look at possible error that Fable identified
- [ ] See what optimizations can be made to increase speed of code
  - [ ] Profile first to ID bottlenecks
- [ ] Try 1999 all-solute correction now that I have p* "start small, ramp up" strategy down (does it still result in unsolvable system of eqns?)
- 


## Unexplored stuff
- What if the composition under diffusion in one phase enters the two phase region. How should this be flagged/handles
  - Could imagine nucleating a new phase (which obviously adds a lot of complexity)
- Add capability for 3 phases to do dissimilar ternary bonding test case
  - Are there any fundamental issues with implementing this (for Illingworth method)?
- Handling of phases disappearing
- Handling of variable molar volumes
  - Can this "easily" be added to the Illingworth models?

<style>
r { color: Red }
o { color: Orange }
y { color: Yellow }
g { color: Green }
b { color: lightskyblue }
pu { color: purple }
</style>