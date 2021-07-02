import torch
import numpy as np
import networkx as nx
import torch_geometric as tg
from model.utils import batch_dihedrals
from model.cycle_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_conformers(data, model):

    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)

    new_pos = torch.zeros([data.batch.size(0), model.n_model_confs, 3])
    dihedral_pairs = model.dihedral_pairs.t().detach().numpy()

    Sx = []
    Sy = []
    in_cycle = 0

    for i, pair in enumerate(dihedral_pairs):

        x_index, y_index = pair
        cycle_added = False
        if in_cycle:
            in_cycle -= 1

        if in_cycle:
            continue

        y_cycle_check = [y_index in cycle for cycle in cycles]
        x_cycle_check = [x_index in cycle for cycle in cycles]

        if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle

            cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x_index)
            cycle_avg_coords, cycle_avg_indices = smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, i) # i instead of i+1

            # new graph
            if x_index not in Sx:
                new_pos[cycle_avg_indices] = cycle_avg_coords
                Sx = []

            else:
                assert sorted(Sx) == Sx
                p_mask = [True if a in Sx else False for a in sorted(cycle_avg_indices)]
                q_mask = [True if a in sorted(cycle_avg_indices) else False for a in Sx]
                p_reorder = sorted(range(len(cycle_avg_indices)), key=lambda k: cycle_avg_indices[k])
                aligned_cycle_coords = align_coords_Kabsch(cycle_avg_coords[p_reorder].permute(1, 0, 2).unsqueeze(0), new_pos[Sx].permute(1, 0, 2), p_mask, q_mask)
                aligned_cycle_coords = aligned_cycle_coords.squeeze(0).permute(1, 0, 2)
                cycle_avg_indices_reordered = [cycle_avg_indices[l] for l in p_reorder]

                # apply to all new coordinates?
                new_pos[cycle_avg_indices_reordered] = aligned_cycle_coords

            Sx.extend(cycle_avg_indices)
            Sx = sorted(list(set(Sx)))
            in_cycle = len(cycle_indices)  # one less than below bc 2 nodes are added to ring
            continue

        if any(y_cycle_check):
            cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y_index)
            cycle_added = True
            in_cycle = len(cycle_indices)+1

        # new graph
        p_coords = torch.zeros([4, model.n_model_confs, 3])
        p_idx = model.neighbors[x_index]

        if x_index not in Sx:
            Sx = []
            # set new immediate neighbor coords for X
            p_coords = model.p_coords[i]
            new_pos[p_idx] = p_coords[0:int(model.dihedral_x_mask[i].sum())]

        else:
            p_coords[0:p_idx.size(0)] = new_pos[p_idx] - new_pos[x_index]

        # update indices
        Sx.extend([x_index])
        Sx.extend(model.neighbors[x_index].detach().numpy())
        Sx = list(set(Sx))

        Sy.extend([y_index])
        Sy.extend(model.neighbors[y_index].detach().numpy())

        # set px
        p_X = new_pos[x_index]

        # translate current Sx
        new_pos_Sx = new_pos[Sx] - p_X

        # set Y
        if cycle_added:
            cycle_avg_coords, cycle_avg_indices = smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, i+1)
            cycle_avg_coords = cycle_avg_coords - cycle_avg_coords[cycle_avg_indices == y_index] # move y to origin
            q_idx = model.neighbors[y_index]
            q_coords_mask = [True if a in q_idx else False for a in cycle_avg_indices]
            q_coords = torch.zeros([4, model.n_model_confs, 3])
            q_reorder = np.argsort([np.where(a == q_idx)[0][0] for a in torch.tensor(cycle_avg_indices)[q_coords_mask]])
            q_coords[0:sum(q_coords_mask)] = cycle_avg_coords[q_coords_mask][q_reorder]
            new_pos_Sy = cycle_avg_coords.clone()
            Sy = cycle_avg_indices

        else:
            q_coords = model.q_coords[i]
            q_idx = model.neighbors[y_index]
            new_pos[q_idx] = q_coords[0:int(model.dihedral_y_mask[i].sum())]
            new_pos[y_index] = torch.zeros_like(p_X)  # q_Y always at the origin
            new_pos_Sy = new_pos[Sy]

        # calculate rotation matrices
        H_XY = rotation_matrix_inf_v2(p_coords, model.x_map_to_neighbor_y[i])
        H_YX = rotation_matrix_inf_v2(q_coords, model.y_map_to_neighbor_x[i])

        # rotate
        new_pos_Sx_2 = torch.matmul(H_XY.unsqueeze(0), new_pos_Sx.unsqueeze(-1)).squeeze(-1)
        new_pos_Sy_2 = torch.matmul(H_YX.unsqueeze(0), new_pos_Sy.unsqueeze(-1)).squeeze(-1)

        # translate q
        new_p_Y = new_pos_Sx_2[Sx == y_index]
        transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
        new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2.unsqueeze(-1)).squeeze(-1) + new_p_Y

        # rotate by gamma
        H_gamma = calculate_gamma(model.n_model_confs, model.dihedral_mask[i], model.c_ij[i], model.v_star[i], Sx, Sy,
                                  p_idx, q_idx, x_index, y_index, new_pos_Sx_2, new_pos_Sy_3, new_p_Y)
        new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2.unsqueeze(-1)).squeeze(-1)

        # update all coordinates
        new_pos[Sy] = new_pos_Sy_3
        new_pos[Sx] = new_pos_Sx_3

        # update indices
        Sx.extend(Sy)
        Sx = sorted(list(set(Sx)))
        Sy = []

    return new_pos


def smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, cycle_start_idx):

    # find index of cycle starting position
    cycle_len = len(cycle_indices)

    # get dihedral pairs corresponding to current cycle
    cycle_pairs = dihedral_pairs[cycle_start_idx:cycle_start_idx+cycle_len]

    # create indices for cycle
    cycle_i = np.arange(cycle_start_idx, cycle_start_idx+cycle_len)

    # create ordered dihedral pairs and indices which each start at a different point in the cycle
    cycle_dihedral_pair_orders = np.stack([np.roll(cycle_pairs, -i, axis=0) for i in range(len(cycle_pairs))])[:-1]
    cycle_i_orders = np.stack([np.roll(cycle_i, -i, axis=0) for i in range(len(cycle_i))])[:-1]

    # intialize lists to track which indices have been added and cycle position vector
    Sx_cycle, Sy_cycle = [[] for i in range(cycle_len)], [[] for i in range(cycle_len)]
    cycle_pos = torch.zeros_like(new_pos).unsqueeze(0).repeat(cycle_len, 1, 1, 1)

    for ii, (pairs, ids) in enumerate(zip(cycle_dihedral_pair_orders, cycle_i_orders)):

        # ii is the enumeration index
        # pairs are the dihedral pairs for the cycle with shape (cycle_len-1, 2)
        # ids are the indices of the dihedral pairs corresponding to the model values with shape (cycle_len-1)

        x_indices, y_indices = pairs.transpose()

        p_coords = torch.zeros([cycle_len, 4, model.n_model_confs, 3])
        p_idx = [model.neighbors[x] for x in x_indices]
        if ii == 0:

            # set new immediate neighbor coords for X
            p_coords = model.p_coords[ids]
            for i, p_i in enumerate(p_idx):
                cycle_pos[i, p_i] = p_coords[i, 0:int(model.dihedral_x_mask[ids[i]].sum())]

        else:
            for i in range(cycle_len):
                p_coords[i, 0:p_idx[i].size(0)] = cycle_pos[i, p_idx[i]] - cycle_pos[i, x_indices[i]]

        # update indices
        q_idx = [model.neighbors[y] for y in y_indices]
        for i, (x_idx, p_idxs) in enumerate(zip(x_indices, p_idx)):

            Sx_cycle[i].extend([x_idx])
            Sx_cycle[i].extend(p_idxs.detach().cpu().numpy())
            Sx_cycle[i] = list(set(Sx_cycle[i]))

            Sy_cycle[i].extend([y_indices[i]])
            Sy_cycle[i].extend(q_idx[i].detach().cpu().numpy())

        # set px
        p_X = cycle_pos[torch.arange(cycle_len), x_indices]

        # translate current Sx
        new_pos_Sx = [cycle_pos[i, Sx_cycle[i]] - p_X[0].unsqueeze(0) for i in range(cycle_len)]

        # set Y
        q_coords = model.q_coords[ids]
        new_pos_Sy = []
        for i, q_i in enumerate(q_idx):
            cycle_pos[i, q_i] = q_coords[i, 0:int(model.dihedral_y_mask[ids[i]].sum())]
            cycle_pos[i, y_indices[i]] = torch.zeros_like(p_X[i])  # q_Y always at the origin
            new_pos_Sy.append(cycle_pos[i, Sy_cycle[i]])

        # calculate rotation matrices
        H_XY = list(map(rotation_matrix_inf_v2, p_coords, model.x_map_to_neighbor_y[ids]))
        H_YX = list(map(rotation_matrix_inf_v2, q_coords, model.y_map_to_neighbor_x[ids]))

        # rotate
        new_pos_Sx_2 = [torch.matmul(H_XY[i].unsqueeze(0), new_pos_Sx[i].unsqueeze(-1)).squeeze(-1) for i in range(cycle_len)]
        new_pos_Sy_2 = [torch.matmul(H_YX[i].unsqueeze(0), new_pos_Sy[i].unsqueeze(-1)).squeeze(-1) for i in range(cycle_len)]

        for i in range(cycle_len):

            # translate q
            new_p_Y = new_pos_Sx_2[i][Sx_cycle[i] == y_indices[i]].squeeze(-1)
            transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
            new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2[i].unsqueeze(-1)).squeeze(-1) + new_p_Y

            # rotate by gamma
            H_gamma = calculate_gamma(model.n_model_confs, model.dihedral_mask[ids[i]], model.c_ij[ids[i]],
                                      model.v_star[ids[i]], Sx_cycle[i], Sy_cycle[i], p_idx[i], q_idx[i], pairs[i][0],
                                      pairs[i][1], new_pos_Sx_2[i], new_pos_Sy_3, new_p_Y)
            new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2[i].unsqueeze(-1)).squeeze(-1)

            # update all coordinates
            cycle_pos[i, Sy_cycle[i]] = new_pos_Sy_3
            cycle_pos[i, Sx_cycle[i]] = new_pos_Sx_3

            # update indices
            Sx_cycle[i].extend(Sy_cycle[i])
            Sx_cycle[i] = list(set(Sx_cycle[i]))

        # update y indices or create mask for last loop
        if not np.all(ids == cycle_i_orders[-1]):
            Sy_cycle = [[] for i in range(cycle_len)]
        else:
            cycle_mask = torch.ones([cycle_pos.size(0), cycle_pos.size(1)])
            for i in range(cycle_len):
                cycle_mask[i, y_indices[i]] = 0
                y_neighbor_ids = model.neighbors[y_indices[i]]
                y_neighbor_ids_not_x = y_neighbor_ids[~model.y_map_to_neighbor_x[ids[i]][0:len(y_neighbor_ids)].bool()]
                cycle_mask[i, y_neighbor_ids_not_x] = 0

    # extract unaligned coords
    final_cycle_coords_unaligned = cycle_pos[:, Sx_cycle[0]]
    q_cycle_coords = final_cycle_coords_unaligned[0].permute(1, 0, 2)  # target
    p_cycle_coords = final_cycle_coords_unaligned[1:].permute(0, 2, 1, 3)  # source

    # align coords with Kabsch algorithm
    q_cycle_coords_aligned = final_cycle_coords_unaligned[0]
    cycle_rmsd_mask = [True if a in np.unique(cycle_pairs) else False for a in Sx_cycle[0]]
    # cycle_rmsd_mask = [True for a in Sx_cycle[0]]
    p_cycle_coords_aligned = align_coords_Kabsch(p_cycle_coords, q_cycle_coords, cycle_rmsd_mask).permute(0, 2, 1, 3)

    # average aligned coords
    cycle_avg_coords_ = torch.vstack([q_cycle_coords_aligned.unsqueeze(0), p_cycle_coords_aligned]) * cycle_mask[:, Sx_cycle[0]].unsqueeze(-1).unsqueeze(-1)
    cycle_avg_coords = cycle_avg_coords_.sum(dim=0) / cycle_mask[:, Sx_cycle[0]].sum(dim=0).unsqueeze(-1).unsqueeze(-1)

    return cycle_avg_coords, Sx_cycle[0]


def construct_conformers_acyclic(data, n_true_confs, n_model_confs, dihedral_pairs, neighbors, model_p_coords,
                                 model_q_coords, dihedral_x_mask, dihedral_y_mask, x_map_to_neighbor_y,
                                 y_map_to_neighbor_x, dihedral_mask, c_ij, v_star):

    pos = torch.cat([torch.cat([p[0][i] for p in data.pos]).unsqueeze(1) for i in range(n_true_confs)], dim=1)
    new_pos = torch.zeros([pos.size(0), n_model_confs, 3]).to(device)
    dihedral_pairs = dihedral_pairs.t().detach().cpu().numpy()

    Sx = []
    Sy = []

    for i, pair in enumerate(dihedral_pairs):

        x_index, y_index = pair

        # skip cycles for now (check if all of y's neighbors are in Sx)
        if np.prod([n in Sx for n in neighbors[y_index]]):
            continue

        # new graph
        p_coords = torch.zeros([4, n_model_confs, 3]).to(device)
        p_idx = neighbors[x_index]

        if x_index not in Sx:
            Sx = []
            # set new immediate neighbor coords for X
            p_coords = model_p_coords[i]
            new_pos[p_idx] = p_coords[0:int(dihedral_x_mask[i].sum())]

        else:
            p_coords[0:p_idx.size(0)] = new_pos[p_idx] - new_pos[x_index]

        # update indices
        Sx.extend([x_index])
        Sx.extend(neighbors[x_index].detach().numpy())
        Sx = list(set(Sx))

        Sy.extend([y_index])
        Sy.extend(neighbors[y_index].detach().numpy())

        # set px
        p_X = new_pos[x_index]

        # translate current Sx
        new_pos_Sx = new_pos[Sx] - p_X

        # set y
        q_coords = model_q_coords[i]
        q_idx = neighbors[y_index]
        new_pos[q_idx] = q_coords[0:int(dihedral_y_mask[i].sum())]
        new_pos[y_index] = torch.zeros_like(p_X)  # q_Y always at the origin
        new_pos_Sy = new_pos[Sy]

        # calculate rotation matrices
        H_XY = rotation_matrix_inf_v2(p_coords, x_map_to_neighbor_y[i])
        H_YX = rotation_matrix_inf_v2(q_coords, y_map_to_neighbor_x[i])

        # rotate
        new_pos_Sx_2 = torch.matmul(H_XY.unsqueeze(0), new_pos_Sx.unsqueeze(-1)).squeeze(-1)
        new_pos_Sy_2 = torch.matmul(H_YX.unsqueeze(0), new_pos_Sy.unsqueeze(-1)).squeeze(-1)

        # translate q
        new_p_Y = new_pos_Sx_2[Sx == y_index]
        transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
        new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2.unsqueeze(-1)).squeeze(-1) + new_p_Y

        # rotate by gamma
        H_gamma = calculate_gamma(n_model_confs, dihedral_mask[i], c_ij[i], v_star[i], Sx, Sy, p_idx, q_idx, x_index,
                                  y_index, new_pos_Sx_2, new_pos_Sy_3, new_p_Y)
        new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2.unsqueeze(-1)).squeeze(-1)

        # update all coordinates
        new_pos[Sy] = new_pos_Sy_3
        new_pos[Sx] = new_pos_Sx_3

        # update indices
        Sx.extend(Sy)
        Sx = sorted(list(set(Sx)))
        Sy = []

    return new_pos


pT_idx, qZ_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
pT_idx = pT_idx.squeeze(-1)
qZ_idx = qZ_idx.squeeze(-1)


def calculate_gamma(n_model_confs, dihedral_mask, c_ij, v_star, Sx, Sy, p_idx, q_idx, x_index, y_index,
                    new_pos_Sx_2, new_pos_Sy_3, new_p_Y):
    # calculate current dihedrals
    pT_prime = torch.zeros([3, n_model_confs, 3]).to(device)
    qZ_translated = torch.zeros([3, n_model_confs, 3]).to(device)

    pY_prime = new_p_Y.repeat(9, 1, 1)
    qX = torch.zeros_like(pY_prime)

    p_ids_in_Sx = [Sx.index(p.item()) for p in p_idx if p.item() != y_index]
    q_ids_in_Sy = [Sy.index(q.item()) for q in q_idx if q.item() != x_index]

    pT_prime[:len(p_ids_in_Sx)] = new_pos_Sx_2[p_ids_in_Sx]
    qZ_translated[:len(q_ids_in_Sy)] = new_pos_Sy_3[q_ids_in_Sy]

    XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos = batch_dihedrals(pT_prime[pT_idx], qX, pY_prime, qZ_translated[qZ_idx])
    A_ij = build_A_matrix_inf(XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos, n_model_confs) * dihedral_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # build A matrix
    A_curr = torch.sum(A_ij * c_ij.unsqueeze(-1), dim=0)
    determinants = torch.det(A_curr) + 1e-10
    A_curr_inv_ = A_curr.view(n_model_confs, 4)[:, [3, 1, 2, 0]] * torch.tensor([[1., -1., -1., 1.]])
    A_curr_inv = (A_curr_inv_ / determinants.unsqueeze(-1)).view(n_model_confs, 2, 2)
    A_curr_inv_v_star = torch.matmul(A_curr_inv, v_star.unsqueeze(-1)).squeeze(-1)

    # get gamma matrix
    v_gamma = A_curr_inv_v_star / (A_curr_inv_v_star.norm(dim=-1, keepdim=True) + 1e-10)
    gamma_cos, gamma_sin = v_gamma.split(1, dim=-1)
    H_gamma = build_gamma_rotation_inf(gamma_sin.squeeze(-1), gamma_cos.squeeze(-1), n_model_confs)

    return H_gamma


def rotation_matrix_inf_v2(neighbor_coords, neighbor_map):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs
        (n_dihedral_pairs, 4, n_generated_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (n_dihedral_pairs, 4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom
        (n_dihedral_pairs, 4) each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
    """

    p_Y = neighbor_coords[neighbor_map.bool(), :].squeeze(0)

    eta_1 = torch.rand_like(p_Y)
    eta_2 = eta_1 - torch.sum(eta_1 * p_Y, dim=-1, keepdim=True) / (torch.linalg.norm(p_Y, dim=-1, keepdim=True)**2 + 1e-10) * p_Y
    eta = eta_2 / torch.linalg.norm(eta_2, dim=-1, keepdim=True)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h3_1 = torch.cross(p_Y, eta, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h2 = -torch.cross(h1, h3, dim=-1)  # (n_dihedral_pairs, n_model_confs, 10)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def build_A_matrix_inf(curr_sin, curr_cos, n_model_confs):

    A_ij = torch.FloatTensor([[[[0, 0], [0, 0]]]]).repeat(9, n_model_confs, 1, 1)
    A_ij[:, :, 0, 0] = curr_cos
    A_ij[:, :, 0, 1] = curr_sin
    A_ij[:, :, 1, 0] = curr_sin
    A_ij[:, :, 1, 1] = -curr_cos

    return A_ij


def build_gamma_rotation_inf(gamma_sin, gamma_cos, n_model_confs):
    H_gamma = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(n_model_confs, 1, 1)
    H_gamma[:, 1, 1] = gamma_cos
    H_gamma[:, 1, 2] = -gamma_sin
    H_gamma[:, 2, 1] = gamma_sin
    H_gamma[:, 2, 2] = gamma_cos

    return H_gamma
