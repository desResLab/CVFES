/* Explicit VMS on GPU kernel programs.
 */
__constant double alpha = 0.58541020;
__constant double beta = 0.13819660;
__constant double w[] = {0.25, 0.25, 0.25, 0.25};
__constant double lN[4][4] = {{0.58541020, 0.13819660, 0.13819660, 0.13819660},
                              {0.13819660, 0.58541020, 0.13819660, 0.13819660},
                              {0.13819660, 0.13819660, 0.58541020, 0.13819660},
                              {0.13819660, 0.13819660, 0.13819660, 0.58541020}};
__constant double lDN[3][4] = {{-1.0, 1.0, 0.0, 0.0},
                               {-1.0, 0.0, 1.0, 0.0},
                               {-1.0, 0.0, 0.0, 1.0}};


/* Initialize the solver, calculate volume, global DN, for each element;
   Assemble the mass matrix.
 */

__kernel void initial_assemble(const long nElms, const long nNodes,
    const __global double *nodes, const __global long *elmNodeIds,
    __global double *volumes, __global double *DNs, __global double *lumpLHS)
    // __global double *dbgNodeCoords, __global double *dbgDetJ)
{
    uint iElm = get_global_id(0);

    long nodeIds[4];
    double nodeCoords[4][3]; // (4,3)

    double jac[3][3];
    double cof[3][3];
    double invJac[3][3];
    double detJ = 0.0;
    double iDetJ = 0.0;

    // Remember element's nodeIds and nodeCoords.
    for (uint i = 0; i < 4; ++i)
    {
        // nodeIds[i] = elmNodeIds[i*nElms+iElm];
        nodeIds[i] = elmNodeIds[iElm*4+i];

        for (uint j = 0; j < 3; ++j)
        {
            // nodeCoords[i][j] = nodes[j*nNodes+nodeIds[i]];
            nodeCoords[i][j] = nodes[nodeIds[i]*3+j];

            // if (iElm == 0)
            // {
            //     dbgNodeCoords[i*3+j] = nodeCoords[i][j];
            // }
        }
    }

    // Calculate jacobian and inverse of jacobian.
    for (uint i = 0; i < 3; ++i)
    {
        for (uint j = 0; j < 3; ++j)
        {
            jac[i][j] = nodeCoords[(j+1)][i] - nodeCoords[0][i];
        }
    }

    // cof[0] = jac[4]*jac[8] - jac[5]*jac[7];
    // cof[1] = jac[5]*jac[6] - jac[3]*jac[8];
    // cof[2] = jac[3]*jac[7] - jac[4]*jac[6];

    // cof[3] = jac[2]*jac[7] - jac[1]*jac[8];
    // cof[4] = jac[0]*jac[8] - jac[2]*jac[6];
    // cof[5] = jac[1]*jac[6] - jac[0]*jac[7];

    // cof[6] = jac[1]*jac[5] - jac[2]*jac[4];
    // cof[7] = jac[2]*jac[3] - jac[0]*jac[5];
    // cof[8] = jac[0]*jac[4] - jac[1]*jac[3];

    // detJ = jac[0]*cof[0] + jac[1]*cof[1] + jac[2]*cof[2];
    // iDetJ = 1.0 / detJ;

    cof[0][0] = jac[1][1]*jac[2][2] - jac[2][1]*jac[1][2];
    cof[0][1] = jac[2][0]*jac[1][2] - jac[1][0]*jac[2][2];
    cof[0][2] = jac[1][0]*jac[2][1] - jac[2][0]*jac[1][1];
    cof[1][0] = jac[2][1]*jac[0][2] - jac[0][1]*jac[2][2];
    cof[1][1] = jac[0][0]*jac[2][2] - jac[2][0]*jac[0][2];
    cof[1][2] = jac[2][0]*jac[0][1] - jac[0][0]*jac[2][1];
    cof[2][0] = jac[0][1]*jac[1][2] - jac[1][1]*jac[0][2];
    cof[2][1] = jac[1][0]*jac[0][2] - jac[0][0]*jac[1][2];
    cof[2][2] = jac[0][0]*jac[1][1] - jac[1][0]*jac[0][1];

    detJ = jac[0][0]*cof[0][0] + jac[0][1]*cof[0][1] + jac[0][2]*cof[0][2];
    iDetJ = 1.0 / detJ;

    // if (iElm == 0)
    // {
    //     dbgDetJ[0] = detJ;
    // }

    for (uint i = 0; i < 3; ++i)
    {
        for (uint j = 0; j < 3; ++j)
        {
            invJac[i][j] = cof[j][i] * iDetJ;
        }
    }

    // 'Assemble' volume and DN.
    volumes[iElm] = detJ / 6.0;

    for (uint i = 0; i < 3; ++i)
    {
        for (uint j = 0; j < 4; ++j)
        {
            DNs[(i*4+j)*nElms+iElm] = lDN[0][j]*invJac[0][i] \
                + lDN[1][j]*invJac[1][i] + lDN[2][j]*invJac[2][i];
        }
    }

    // Assemble lumped mass matrix, e.g. +volume/4.0 to each vertex.
    for (uint i = 0; i < 4; ++i) // 4 vertices (nodes)
    {
        for (uint j = 0; j < 4; ++j) // 4 d.o.f.s
        {
            lumpLHS[j*nNodes+nodeIds[i]] += detJ / 24.0;
        }
    }
}


__kernel void assemble_RHS(const long nElms, const long nNodes,
    const __global long *elmNodeIds, const __global double *fs,
    const __global double *volumes, const __global double *DNs,
    const __global double *duP, const __global double *preDuP,
    const __global double *sdus, const __global double *params,
    __global double *RHS)
{
    uint iElm = get_global_id(0);

    long nodeIds[4];
    double Ve;
    double DN[3][4]; // 3*4
    
    double f[4][3];
    double sdu[4][3];
    double hdu[4][3]; // 4*3
    double hp[4];
    double ha[4][3];
    double gradHdu[3][3]; // 3*3

    double wGp;
    double hah[3];
    double hph;
    double sduh[3];
    double fh[3];

    double trGradHdu = 0.0;
    double ahGradHu[3];
    double ahDN;
    double sduhDN;
    double lRes[4][4];

    // parameters
    double nu = params[2];
    double invEpsilon = params[4];

    // Memory clear first.
    for (uint i = 0; i < 4; ++i)
    {
        for (uint j = 0; j < 4; ++j)
        {
            lRes[i][j] = 0.0;
        }
    }

    // Remember element's nodeIds.
    for (uint i = 0; i < 4; ++i)
    {
        // nodeIds[i] = elmNodeIds[i*nElms+iElm];
        nodeIds[i] = elmNodeIds[iElm*4+i];

        // Calculate initial values.
        for (uint j = 0; j < 3; ++j)
        {
            sdu[i][j] = sdus[(i*3+j)*nElms+iElm];
            f[i][j] = fs[j*nNodes+nodeIds[i]];
            
            hdu[i][j] = 1.5*duP[j*nNodes+nodeIds[i]] - 0.5*preDuP[j*nNodes+nodeIds[i]];
            ha[i][j] = hdu[i][j] + sdu[i][j];
        }
        hp[i] = 1.5*duP[3*nNodes+nodeIds[i]] - 0.5*preDuP[3*nNodes+nodeIds[i]];
    }

    Ve = volumes[iElm];
    for (uint i = 0; i < 3; ++i)
    {
        for (uint j = 0; j < 4; ++j)
        {
            DN[i][j] = DNs[(i*4+j)*nElms+iElm];
        }
    }

    // Calculate elemental values.
    for (uint i = 0; i < 3; ++i) // partial u_x, u_y, u_z
    {
        for (uint j = 0; j < 3; ++j) // partial x, y, z
        {
            gradHdu[i][j] = DN[j][0]*hdu[0][i] + DN[j][1]*hdu[1][i] \
                          + DN[j][2]*hdu[2][i] + DN[j][3]*hdu[3][i];
        }

        trGradHdu += gradHdu[i][i];
    }

    // Loop through Gaussian points to do numerical integration.
    for (uint iGp = 0; iGp < 4; ++iGp)
    {
        wGp = w[iGp] * Ve;

        // Calculate values at Gaussian point iGp.
        for (uint i = 0; i < 3; ++i)
        {
            hah[i] = 0.0;
            sduh[i] = 0.0;
            fh[i] = 0.0;

            for (uint j = 0; j < 4; ++j)
            {
                hah[i] += ha[j][i] * lN[iGp][j];
                sduh[i] += sdu[j][i] * lN[iGp][j];
                fh[i] += f[j][i] * lN[iGp][j];
            }
        }

        hph = 0.0;
        for (uint i = 0; i < 4; ++i)
        {
            hph += hp[i] * lN[iGp][i];
        }

        // Calculate medium values.
        for (uint i = 0; i < 3; ++i)
        {
            ahGradHu[i] = 0.0;

            for (uint j = 0; j < 3; ++j)
            {
                ahGradHu[i] += gradHdu[i][j] * hah[j];
            }
        }


        // Assemble for each point.
        for (uint a = 0; a < 4; ++a)
        {
            // Calculate temporary value first.
            ahDN = 0.0;
            sduhDN = 0.0;
            for (uint i = 0; i < 3; ++i)
            {
                ahDN += hah[i] * DN[i][a];
                sduhDN += sduh[i] * DN[i][a];
            }
            
            // Assemble first 3 d.o.f.s.
            for (uint i = 0; i < 3; ++i)
            {
                for (uint j = 0; j < 3; ++j)
                {
                    lRes[a][i] += wGp*nu*gradHdu[i][j]*DN[j][a];
                }

                lRes[a][i] += wGp*(ahGradHu[i]*lN[iGp][a] \
                            - hph*DN[i][a] - sduh[i]*ahDN \
                            - fh[i]*lN[iGp][a]);
                
                // lRes[a][i] += wGp*ahGradHu[i]*lN[iGp][a];
            }

            // Assemble last d.o.f. for pressure.
            lRes[a][3] += wGp*(trGradHdu*lN[iGp][a] - sduhDN)*invEpsilon;
        }
    }

    // Assemble to global RHS.
    for (uint a = 0; a < 4; ++a)
    {
        for (uint i = 0; i < 4; ++i)
        {
            RHS[i*nNodes+nodeIds[a]] += lRes[a][i];
        }
    }
}


__kernel void calc_res(const long nGrps, const long ndof, const double dt,
    const __global double *RHS, const __global double *lumpLHS,
    const __global double *preDuP, __global double *duP)
{
    uint lclSize = get_local_size(0);
    uint j = get_local_id(0);
    uint idx;

    for (uint i = get_group_id(0); i < nGrps; i += get_num_groups(0))
    {
        idx = i * lclSize + j;
        
        if (idx < ndof)
        {
            duP[idx] = preDuP[idx] - dt * RHS[idx] / lumpLHS[idx];
        }
    }
}


__kernel void apply_drchBC(const long nNodes,
    const __global double *drchValues, const __global long *drchIndices,
    __global double *duP)
{
    uint idx = get_global_id(0);
    uint nSize = get_global_size(0);

    for (uint i = 0; i < 3; ++i)
    {
        duP[i*nNodes+drchIndices[idx]] = drchValues[i*nSize+idx];
    }
}


__kernel void test_memory_layout(const long nRow, const long nColumns,
    __global double *testMemory)
{
    uint iElm = get_global_id(0);

    for (uint i = 0; i < nRow; ++i)
    {
        testMemory[i*nColumns+iElm] = i;
    }
}














