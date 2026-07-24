.title FIVE TRANSISTOR COMPARATOR
************************5T comparator 0.35um****************************
.param W0=20u   W1=80u   W2=80u   W3=40u   W4=40u
.param L0=0.35u

************************* comparator definition *************************
.subckt cmp vinp vinn vout vdd gnd

***********************
* 5T structure:
* M1/M2: NMOS differential pair
* M3/M4: PMOS current mirror load
* M5: tail current source
***********************

* NMOS differential pair
M1   n1   vinp   ntail   gnd   nmos_3p3  L=L0 W=W1
M2   vout   vinn   ntail   gnd   nmos_3p3  L=L0 W=W2

* PMOS current mirror load
M3   n1   n1     vdd     vdd   pmos_3p3  L=L0 W=W3
M4   vout n1     vdd     vdd   pmos_3p3  L=L0 W=W4

***********************
* Ideal tail current source
***********************
I_tail ntail gnd DC 100u

.ends cmp
***************************end definition*****************************