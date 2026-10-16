# Illingworth–Golosnoy moving-boundary code (MAP_KINETICS_MOVING_BOUNDARY)

This folder holds the C++ reference implementation of the conservative implicit
front-fixing (IFF) method. kawin's Illingworth solvers are checked against it:
`examples/Illingworth2005/compare_illingworth2005_planar.py` and the tests that
call it compile these files with `g++`.

## Acknowledgement

- **Source:** Materials Algorithms Project (MAP), Department of Materials Science
  and Metallurgy, University of Cambridge. The entry is MAP_KINETICS_MOVING_BOUNDARY:
  <https://www.phase-trans.msm.cam.ac.uk/map/kinetics/programs/ConservativeIFF.html>.
  The download is <https://www.phase-trans.msm.cam.ac.uk/map/kinetics/tar/illingworth.zip>.
- **Code originators:** T.C. Illingworth and I.O. Golosnoy, University of Cambridge.
  The code was added to MAP in February 2005.
- **Method:** T.C. Illingworth, I.O. Golosnoy, "Numerical solutions of
  diffusion-controlled moving boundary problems which conserve solute", Journal
  of Computational Physics 209 (2005) 207–225, doi:10.1016/j.jcp.2005.02.031.

## Terms of use

These files are **not** covered by kawin's MIT licence. They are redistributed
under the conditions MAP sets for its code
(<https://www.phase-trans.msm.cam.ac.uk/map/donor/advice.html>):

> Code downloaded from the MAP website may be modified and distributed without
> restriction PROVIDED that:
> - MAP and the code originator are acknowledged in the documentation and in the
>   body of the source code
> - Any modifications are clearly indicated as such
> - No attempt is made to charge for subsequent distribution of the code

MAP's description of the project (S. Cardie and H.K.D.H. Bhadeshia,
<https://www.phase-trans.msm.cam.ac.uk/map/pubs/graz.html>) also states that the
software is in the public domain. MAP provides its programs "AS IS", without
warranty (<https://www.phase-trans.msm.cam.ac.uk/map/entry.html>).

## Changes from the MAP download

- **Source files:** each `.cpp` and `.h` file starts with an added comment block
  acknowledging MAP and the authors and stating these terms. Nothing else was
  changed; below that block every file is identical to the MAP download. The
  files are committed with LF line endings, as in the download; Git may check
  them out with CRLF on Windows.
- **`ConservativeIFF.html`:** MAP's documentation page for the entry, unchanged.
- **`IllingworthGolosnoy.pdf`:** MAP's zip also contains the authors' manuscript
  of the paper. It is not needed to build or test the code and is not committed;
  it is excluded in `.gitignore`. Download it from MAP, or see the published
  paper.
