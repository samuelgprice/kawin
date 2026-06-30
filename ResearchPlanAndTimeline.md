


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
                    - Seems that initial mass will be off because the interface compositions are given a half-cell of space (same issue that I had with Lee and Oh stuff)
                - [ ] check the mass conservation by doing control volume sum (likely in "_lfdf_phase_update")
                - [ ] Could email authors with questions (e.g., what dx/dt was used to make figures)


    - compare results to those of Lee binary
- then implement ternary version
    - compare results to Lee ternary


## Lee work: 1996 and 1999 papers:

- <r> Maybe try totally different way of integrating composition profile </r>:
    - instead of using trapezoid maybe just do composition times cell width (akin to finite volume approach)


<style>
r { color: Red }
o { color: Orange }
y { color: Yellow }
g { color: Green }
b { color: lightskyblue }
pu { color: purple }
</style>