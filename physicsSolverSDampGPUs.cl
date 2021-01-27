
void TtKT(const double *T, const double *K,
          double *tmp, double *res)
{
    for (uint i = 0; i < 3; i++)
    {
        for (uint j = 0; j < 3; j++)
        {
            tmp[i*3+j] = T[i]*K[j] + T[3+i]*K[3+j] + T[6+i]*K[6+j];
        }
    }

    for (uint i = 0; i < 3; i++)
    {
        for (uint j = 0; j < 3; j++)
        {
            res[i*3+j] = tmp[i*3]*T[j] + tmp[i*3+1]*T[3+j] + tmp[i*3+2]*T[6+j];
        }
    }
}

/* Assemble matrix K (stiffness) and M (mass) at the initial.
   - nodes: (nGNodes, 3)    the global coords of all the nodes.
   - elmNodeIds: (nElm, 3)  the node ids' of each element.
   - Ku: (nGNodes*3, nSmp)  the product result after assembling.
 */

__kernel void assemble_K_M_P(const long nSmp, const double pressure,
                             const __global double *pVals, const __global double *nodes,
                             const __global long *elmNodeIds, const __global double *thickness,
                             const __global double *elmThicknessE,
                             const __global double *u, __global double *Ku,
                             __global double *LM, __global double *P)
{
    uint iElm = get_global_id(0);

    const __global long *nodeIds = elmNodeIds + iElm * 3;
    __global double *sM;
    __global double *sKu;
    __global double *sP;

    const double density = pVals[0];
    // pE[0]: v
    // pE[1]: 0.5*(1-v)
    // pE[2]: 0.5*k*(1-v)
    // pE[3]: (1-v^2)
    const __global double *pE = pVals + 1;

    double elmThick[3];
    double T[9];
    double lNodes[6];
    double lB[6];
    double llK[9] = {0};
    double glK[9];
    // tmp vars
    double tmpVec[3];
    double tmpBP[4]; // product of two rows of B
    double tmpMat[9];

    double area = 0.0;
    double lm = 0.0;
    // tmp vars
    double tmpVal = 0.0;
    double tmpNorm = 0.0;

    // Transform coordernates to referenced coord.
    // 1. Get coordinate transformation matrix T.
    // T : x 0 1 2
    //     y 3 4 5
    //     z 6 7 8
    // x
    tmpVec[0] = nodes[nodeIds[2]*3] - nodes[nodeIds[1]*3];
    tmpVec[1] = nodes[nodeIds[2]*3+1] - nodes[nodeIds[1]*3+1];
    tmpVec[2] = nodes[nodeIds[2]*3+2] - nodes[nodeIds[1]*3+2];
    tmpNorm = sqrt(pow(tmpVec[0],2) + pow(tmpVec[1],2) + pow(tmpVec[2],2));
    T[0] = tmpVec[0] / tmpNorm;
    T[1] = tmpVec[1] / tmpNorm;
    T[2] = tmpVec[2] / tmpNorm;
    // y
    tmpVec[0] = nodes[nodeIds[0]*3] - nodes[nodeIds[2]*3];
    tmpVec[1] = nodes[nodeIds[0]*3+1] - nodes[nodeIds[2]*3+1];
    tmpVec[2] = nodes[nodeIds[0]*3+2] - nodes[nodeIds[2]*3+2];
    tmpVal = T[0]*tmpVec[0] + T[1]*tmpVec[1] + T[2]*tmpVec[2];
    tmpVec[0] -= tmpVal*T[0];
    tmpVec[1] -= tmpVal*T[1];
    tmpVec[2] -= tmpVal*T[2];
    tmpNorm = sqrt(pow(tmpVec[0],2) + pow(tmpVec[1],2) + pow(tmpVec[2],2));
    T[3] = tmpVec[0] / tmpNorm;
    T[4] = tmpVec[1] / tmpNorm;
    T[5] = tmpVec[2] / tmpNorm;
    // z, the cross product of x and y
    T[6] = T[1]*T[5] - T[2]*T[4];
    T[7] = T[2]*T[3] - T[0]*T[5];
    T[8] = T[0]*T[4] - T[1]*T[3];

    // 2. Transform triangle in 3D to local 2D coord.
    // node 0 : 0 1     0 1
    // node 1 : 0 1     2 3
    // node 2 : 0 1     4 5
    lNodes[0] = nodes[nodeIds[0]*3]*T[0] + nodes[nodeIds[0]*3+1]*T[1] + nodes[nodeIds[0]*3+2]*T[2];
    lNodes[1] = nodes[nodeIds[0]*3]*T[3] + nodes[nodeIds[0]*3+1]*T[4] + nodes[nodeIds[0]*3+2]*T[5];
    lNodes[2] = nodes[nodeIds[1]*3]*T[0] + nodes[nodeIds[1]*3+1]*T[1] + nodes[nodeIds[1]*3+2]*T[2];
    lNodes[3] = nodes[nodeIds[1]*3]*T[3] + nodes[nodeIds[1]*3+1]*T[4] + nodes[nodeIds[1]*3+2]*T[5];
    lNodes[4] = nodes[nodeIds[2]*3]*T[0] + nodes[nodeIds[2]*3+1]*T[1] + nodes[nodeIds[2]*3+2]*T[2];
    lNodes[5] = nodes[nodeIds[2]*3]*T[3] + nodes[nodeIds[2]*3+1]*T[4] + nodes[nodeIds[2]*3+2]*T[5];

    // 3. Calculate area by     |x1  y1  1|     lNodes[0] lNodes[1] 1
    //                      0.5*|x2  y2  1|     lNodes[2] lNodes[3] 1
    //                          |x3  y3  1|     lNodes[4] lNodes[5] 1
    area = 0.5*(lNodes[0]*(lNodes[3]-lNodes[5]) - lNodes[2]*(lNodes[1]-lNodes[5]) + lNodes[4]*(lNodes[1]-lNodes[3]));

    // 4. Compute local B (strain matrix).
    lB[0] = lNodes[3] - lNodes[5]; // y23: y2 - y3
    lB[1] = lNodes[4] - lNodes[2]; // x32: x3 - x2
    lB[2] = lNodes[5] - lNodes[1]; // y31: y3 - y1
    lB[3] = lNodes[0] - lNodes[4]; // x13: x1 - x3
    lB[4] = lNodes[1] - lNodes[3]; // y12: y1 - y2
    lB[5] = lNodes[2] - lNodes[0]; // x21: x2 - x1


    for (uint iSmp = 0; iSmp < nSmp; iSmp++)
    {
        sKu = Ku + iSmp;
        sM = LM + iSmp;
        sP = P + iSmp;

        elmThick[0] = thickness[nodeIds[0]*nSmp+iSmp];
        elmThick[1] = thickness[nodeIds[1]*nSmp+iSmp];
        elmThick[2] = thickness[nodeIds[2]*nSmp+iSmp];

        // Loop through the nodes and "assemble" K
        // 5. Compute local K (stiffness) matrix.
        tmpVal = elmThicknessE[iElm*nSmp+iSmp]/(4.0*area*pE[3]);
        for (uint i = 0; i < 3; i++)
        {
            for (uint j = i; j < 3; j++)
            {
                tmpBP[0] = lB[i*2]*lB[j*2];
                tmpBP[1] = lB[i*2]*lB[j*2+1];
                tmpBP[2] = lB[i*2+1]*lB[j*2];
                tmpBP[3] = lB[i*2+1]*lB[j*2+1];

                llK[0] = tmpBP[0] + tmpBP[3]*pE[1];
                llK[1] = tmpBP[1]*pE[0] + tmpBP[2]*pE[1];
                llK[3] = tmpBP[2]*pE[0] + tmpBP[1]*pE[1];
                llK[4] = tmpBP[3] + tmpBP[0]*pE[1];
                llK[8] = (tmpBP[0] + tmpBP[3])*pE[2];

                // 6. Transform to global coord, gK = TtKT.
                TtKT(T, llK, tmpMat, glK);

                // 7. Calculate Ku, then assemble to vector 'Ku'.
                sKu[nodeIds[i]*3*nSmp] += (glK[0]*u[nodeIds[j]*3] + glK[1]*u[nodeIds[j]*3+1] + glK[2]*u[nodeIds[j]*3+2])*tmpVal;
                sKu[(nodeIds[i]*3+1)*nSmp] += (glK[3]*u[nodeIds[j]*3] + glK[4]*u[nodeIds[j]*3+1] + glK[5]*u[nodeIds[j]*3+2])*tmpVal;
                sKu[(nodeIds[i]*3+2)*nSmp] += (glK[6]*u[nodeIds[j]*3] + glK[7]*u[nodeIds[j]*3+1] + glK[8]*u[nodeIds[j]*3+2])*tmpVal;
                // use symetrical char. to save computation
                if (j > i)
                {
                    sKu[nodeIds[j]*3*nSmp] += (glK[0]*u[nodeIds[i]*3] + glK[3]*u[nodeIds[i]*3+1] + glK[6]*u[nodeIds[i]*3+2])*tmpVal;
                    sKu[(nodeIds[j]*3+1)*nSmp] += (glK[1]*u[nodeIds[i]*3] + glK[4]*u[nodeIds[i]*3+1] + glK[7]*u[nodeIds[i]*3+2])*tmpVal;
                    sKu[(nodeIds[j]*3+2)*nSmp] += (glK[2]*u[nodeIds[i]*3] + glK[5]*u[nodeIds[i]*3+1] + glK[8]*u[nodeIds[i]*3+2])*tmpVal;
                }
            }

            // 'Assemble' the force vector.
            sP[nodeIds[i]*3*nSmp] += T[6] * pressure * area / 3.0;
            sP[(nodeIds[i]*3+1)*nSmp] += T[7] * pressure * area / 3.0;
            sP[(nodeIds[i]*3+2)*nSmp] += T[8] * pressure * area / 3.0;
        }


        // Calculate M_ab^e
        // ab = 00
        lm = density * area * (elmThick[0]/4.0 + elmThick[1]/8.0 + elmThick[2]/8.0) / 3.0;
        sM[(nodeIds[0]*3)*nSmp] += lm;
        sM[(nodeIds[0]*3+1)*nSmp] += lm;
        sM[(nodeIds[0]*3+2)*nSmp] += lm;

        // ab = 01
        lm = density * area * (elmThick[0]/8.0 + elmThick[1]/8.0) / 3.0;
        sM[(nodeIds[0]*3)*nSmp] += lm;
        sM[(nodeIds[0]*3+1)*nSmp] += lm;
        sM[(nodeIds[0]*3+2)*nSmp] += lm;
        // ab = 10 with symetrical
        sM[(nodeIds[1]*3)*nSmp] += lm;
        sM[(nodeIds[1]*3+1)*nSmp] += lm;
        sM[(nodeIds[1]*3+2)*nSmp] += lm;

        // ab = 02
        lm = density * area * (elmThick[0]/8.0 + elmThick[2]/8.0) / 3.0;
        sM[(nodeIds[0]*3)*nSmp] += lm;
        sM[(nodeIds[0]*3+1)*nSmp] += lm;
        sM[(nodeIds[0]*3+2)*nSmp] += lm;
        // ab = 20 with symetrical
        sM[(nodeIds[2]*3)*nSmp] += lm;
        sM[(nodeIds[2]*3+1)*nSmp] += lm;
        sM[(nodeIds[2]*3+2)*nSmp] += lm;

        // Row 1
        // ab = 11
        lm = density * area * (elmThick[0]/8.0 + elmThick[1]/4.0 + elmThick[2]/8.0) / 3.0;
        sM[(nodeIds[1]*3)*nSmp] += lm;
        sM[(nodeIds[1]*3+1)*nSmp] += lm;
        sM[(nodeIds[1]*3+2)*nSmp] += lm;

        // ab = 12
        lm = density * area * (elmThick[1]/8.0 + elmThick[2]/8.0) / 3.0;
        sM[(nodeIds[1]*3)*nSmp] += lm;
        sM[(nodeIds[1]*3+1)*nSmp] += lm;
        sM[(nodeIds[1]*3+2)*nSmp] += lm;
        // ab = 21 with symetrical
        sM[(nodeIds[2]*3)*nSmp] += lm;
        sM[(nodeIds[2]*3+1)*nSmp] += lm;
        sM[(nodeIds[2]*3+2)*nSmp] += lm;

        // Row 2
        // ab = 22
        lm = density * area * (elmThick[0]/8.0 + elmThick[1]/4.0 + elmThick[2]/8.0) / 3.0;
        sM[(nodeIds[2]*3)*nSmp] += lm;
        sM[(nodeIds[2]*3+1)*nSmp] += lm;
        sM[(nodeIds[2]*3+2)*nSmp] += lm;
    }
}


/* Assemble matrix K (stiffness) at each time step.
   - nodes: (nGNodes, 3)    the global coords of all the nodes.
   - elmNodeIds: (nElm, 3)  the node ids' of each element.
   - u: (nGNodes*3, nSmp)   the displacement of all samples at each dof.
 */

__kernel void assemble_K_P(const long nElms, const long nSmp, const double pressure,
                           const __global double *pVals, const __global double *nodes,
                           const __global long *elmNodeIds, const __global double *elmThicknessE,
                           const __global double *u, __global double *Ku, __global double *P)
{
    for (uint iElm = get_group_id(0); iElm < nElms; iElm += get_num_groups(0))
    {
        const __global long *nodeIds = elmNodeIds + iElm * 3;

        const double density = pVals[0];
        // pE[0]: v
        // pE[1]: 0.5*(1-v)
        // pE[2]: 0.5*k*(1-v)
        // pE[3]: (1-v^2)
        const __global double *pE = pVals + 1;

        double sNodes[9];
        double T[9];
        double lNodes[6];
        double lB[6];
        double llK[9] = {0};
        double glK[9];
        // tmp vars
        double tmpVec[3];
        double tmpBP[4]; // product of two rows of B
        double tmpMat[9];

        double area = 0.0;
        // tmp vars
        double tmpVal = 0.0;
        double tmpNorm = 0.0;

        for (uint iSmp = get_local_id(0); iSmp < nSmp; iSmp += get_local_size(0))
        {
            const __global double *su = u + iSmp;
            __global double *sKu = Ku + iSmp;
            __global double *sP = P + iSmp;

            // Get the updated coordinates.
            // node 0
            sNodes[0] = nodes[nodeIds[0]*3] + su[(nodeIds[0]*3)*nSmp];
            sNodes[1] = nodes[nodeIds[0]*3+1] + su[(nodeIds[0]*3+1)*nSmp];
            sNodes[2] = nodes[nodeIds[0]*3+2] + su[(nodeIds[0]*3+2)*nSmp];
            // node 1
            sNodes[3] = nodes[nodeIds[1]*3] + su[(nodeIds[1]*3)*nSmp];
            sNodes[4] = nodes[nodeIds[1]*3+1] + su[(nodeIds[1]*3+1)*nSmp];
            sNodes[5] = nodes[nodeIds[1]*3+2] + su[(nodeIds[1]*3+2)*nSmp];
            // node 2
            sNodes[6] = nodes[nodeIds[2]*3] + su[(nodeIds[2]*3)*nSmp];
            sNodes[7] = nodes[nodeIds[2]*3+1] + su[(nodeIds[2]*3+1)*nSmp];
            sNodes[8] = nodes[nodeIds[2]*3+2] + su[(nodeIds[2]*3+2)*nSmp];

            // Transform coordernates to referenced coord.
            // 1. Get coordinate transformation matrix T.
            // T : x 0 1 2
            //     y 3 4 5
            //     z 6 7 8
            // x
            tmpVec[0] = sNodes[6] - sNodes[3];
            tmpVec[1] = sNodes[7] - sNodes[4];
            tmpVec[2] = sNodes[8] - sNodes[5];
            tmpNorm = sqrt(pow(tmpVec[0],2) + pow(tmpVec[1],2) + pow(tmpVec[2],2));
            T[0] = tmpVec[0] / tmpNorm;
            T[1] = tmpVec[1] / tmpNorm;
            T[2] = tmpVec[2] / tmpNorm;
            // y
            tmpVec[0] = sNodes[0] - sNodes[6];
            tmpVec[1] = sNodes[1] - sNodes[7];
            tmpVec[2] = sNodes[2] - sNodes[8];
            tmpVal = T[0]*tmpVec[0] + T[1]*tmpVec[1] + T[2]*tmpVec[2];
            tmpVec[0] -= tmpVal*T[0];
            tmpVec[1] -= tmpVal*T[1];
            tmpVec[2] -= tmpVal*T[2];
            tmpNorm = sqrt(pow(tmpVec[0],2) + pow(tmpVec[1],2) + pow(tmpVec[2],2));
            T[3] = tmpVec[0] / tmpNorm;
            T[4] = tmpVec[1] / tmpNorm;
            T[5] = tmpVec[2] / tmpNorm;
            // z, the cross product of x and y
            T[6] = T[1]*T[5] - T[2]*T[4];
            T[7] = T[2]*T[3] - T[0]*T[5];
            T[8] = T[0]*T[4] - T[1]*T[3];

            // 2. Transform triangle in 3D to local 2D coord.
            // node 0 : 0 1     0 1
            // node 1 : 0 1     2 3
            // node 2 : 0 1     4 5
            lNodes[0] = sNodes[0]*T[0] + sNodes[1]*T[1] + sNodes[2]*T[2];
            lNodes[1] = sNodes[0]*T[3] + sNodes[1]*T[4] + sNodes[2]*T[5];
            lNodes[2] = sNodes[3]*T[0] + sNodes[4]*T[1] + sNodes[5]*T[2];
            lNodes[3] = sNodes[3]*T[3] + sNodes[4]*T[4] + sNodes[5]*T[5];
            lNodes[4] = sNodes[6]*T[0] + sNodes[7]*T[1] + sNodes[8]*T[2];
            lNodes[5] = sNodes[6]*T[3] + sNodes[7]*T[4] + sNodes[8]*T[5];

            // 3. Calculate area by     |x1  y1  1|     lNodes[0] lNodes[1] 1
            //                      0.5*|x2  y2  1|     lNodes[2] lNodes[3] 1
            //                          |x3  y3  1|     lNodes[4] lNodes[5] 1
            area = 0.5*(lNodes[0]*(lNodes[3]-lNodes[5]) - lNodes[2]*(lNodes[1]-lNodes[5]) + lNodes[4]*(lNodes[1]-lNodes[3]));

            // 4. Compute local B (strain matrix).
            lB[0] = lNodes[3] - lNodes[5]; // y23: y2 - y3
            lB[1] = lNodes[4] - lNodes[2]; // x32: x3 - x2
            lB[2] = lNodes[5] - lNodes[1]; // y31: y3 - y1
            lB[3] = lNodes[0] - lNodes[4]; // x13: x1 - x3
            lB[4] = lNodes[1] - lNodes[3]; // y12: y1 - y2
            lB[5] = lNodes[2] - lNodes[0]; // x21: x2 - x1

            // Loop through the nodes and "assemble" K
            // 5. Compute local K (stiffness) matrix.
            tmpVal = elmThicknessE[iElm*nSmp+iSmp]/(4.0*area*pE[3]);
            for (uint i = 0; i < 3; i++)
            {
                for (uint j = i; j < 3; j++)
                {
                    tmpBP[0] = lB[i*2]*lB[j*2];
                    tmpBP[1] = lB[i*2]*lB[j*2+1];
                    tmpBP[2] = lB[i*2+1]*lB[j*2];
                    tmpBP[3] = lB[i*2+1]*lB[j*2+1];

                    llK[0] = tmpBP[0] + tmpBP[3]*pE[1];
                    llK[1] = tmpBP[1]*pE[0] + tmpBP[2]*pE[1];
                    llK[3] = tmpBP[2]*pE[0] + tmpBP[1]*pE[1];
                    llK[4] = tmpBP[3] + tmpBP[0]*pE[1];
                    llK[8] = (tmpBP[0] + tmpBP[3])*pE[2];

                    // 6. Transform to global coord, gK = TtKT.
                    TtKT(T, llK, tmpMat, glK);

                    // 7. Calculate Ku, then assemble to vector 'Ku'.
                    sKu[(nodeIds[i]*3)*nSmp] += (glK[0]*su[(nodeIds[j]*3)*nSmp] \
                                                 + glK[1]*su[(nodeIds[j]*3+1)*nSmp] \
                                                 + glK[2]*su[(nodeIds[j]*3+2)*nSmp])*tmpVal;
                    sKu[(nodeIds[i]*3+1)*nSmp] += (glK[3]*su[(nodeIds[j]*3)*nSmp] \
                                                   + glK[4]*su[(nodeIds[j]*3+1)*nSmp] \
                                                   + glK[5]*su[(nodeIds[j]*3+2)*nSmp])*tmpVal;
                    sKu[(nodeIds[i]*3+2)*nSmp] += (glK[6]*su[(nodeIds[j]*3)*nSmp] \
                                                   + glK[7]*su[(nodeIds[j]*3+1)*nSmp] \
                                                   + glK[8]*su[(nodeIds[j]*3+2)*nSmp])*tmpVal;
                    // use symetrical char. to save computation
                    if (j > i)
                    {
                        sKu[(nodeIds[j]*3)*nSmp] += (glK[0]*su[(nodeIds[i]*3)*nSmp] \
                                                     + glK[3]*su[(nodeIds[i]*3+1)*nSmp] \
                                                     + glK[6]*su[(nodeIds[i]*3+2)*nSmp])*tmpVal;
                        sKu[(nodeIds[j]*3+1)*nSmp] += (glK[1]*su[(nodeIds[i]*3)*nSmp] \
                                                       + glK[4]*su[(nodeIds[i]*3+1)*nSmp] \
                                                       + glK[7]*su[(nodeIds[i]*3+2)*nSmp])*tmpVal;
                        sKu[(nodeIds[j]*3+2)*nSmp] += (glK[2]*su[(nodeIds[i]*3)*nSmp] \
                                                       + glK[5]*su[(nodeIds[i]*3+1)*nSmp] \
                                                       + glK[8]*su[(nodeIds[i]*3+2)*nSmp])*tmpVal;
                    }
                }

                // 'Assemble' the force vector.
                sP[nodeIds[i]*3*nSmp] += T[6] * pressure * area / 3.0;
                sP[(nodeIds[i]*3+1)*nSmp] += T[7] * pressure * area / 3.0;
                sP[(nodeIds[i]*3+2)*nSmp] += T[8] * pressure * area / 3.0;
            }
        }
    }
}


/* Calculate acceleration ddu at the initial.
   - each group loop through columns (nSmp)
 */

__kernel void calc_ddu(const long nSmp, const long ndof,
                       const __global double *P0, const __global double *Ku,
                       const __global double *LHS, __global double *ddu)
{
    for (uint i = get_group_id(0); i < ndof; i += get_num_groups(0))
    {
        for (uint j = get_local_id(0); j < nSmp; j += get_local_size(0))
        {
            ddu[i*nSmp+j] = (P0[i*nSmp+j] - Ku[i*nSmp+j]) / LHS[i*nSmp+j];
        }
    }
}


/* Calculate u of next time step.
   - each group loop through columns (nSmp)
 */

__kernel void calc_u(const long nSmp, const long ndof, const double dt, const double damp,
                       const __global double *P, const __global double *Ku,
                       const __global double *LM, const __global double *LHS,
                       const __global double *u, const __global double *up,
                       __global double *ures)
{
    for (uint i = get_group_id(0); i < ndof; i += get_num_groups(0))
    {
        for (uint j = get_local_id(0); j < nSmp; j += get_local_size(0))
        {
            ures[i*nSmp+j] = (dt*dt*((P[i*nSmp+j]-damp*(u[i*nSmp+j]-up[i*nSmp+j])/dt) - Ku[i*nSmp+j]) + LM[i*nSmp+j]*(2.0*u[i*nSmp+j] - up[i*nSmp+j])) / LHS[i*nSmp+j];
        }
    }
}


/* Calculate u of next time step, add on the appTrac
   - each group loop through columns (nSmp)
 */

__kernel void calc_u_appTrac(const long nSmp, const long ndof, const double dt,
                             const __global double *LHS, const __global double *appTrac,
                             __global double *ures)
{
    for (uint i = get_group_id(0); i < ndof; i += get_num_groups(0))
    {
        for (uint j = get_local_id(0); j < nSmp; j += get_local_size(0))
        {
            ures[i*nSmp+j] += dt*dt*appTrac[i] / LHS[i*nSmp+j];
        }
    }
}


