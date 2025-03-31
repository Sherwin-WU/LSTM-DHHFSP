#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 12:04
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : NSGA_II.py
# @Statement : Nondominated Sorting Genetic Algorithm II (NSGA-II)
# @Reference : Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197.
import numpy as np
#import matplotlib.pyplot as plt
import function as fc

def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def selection(pop, rank, cd, pc):
    # improved binary tournament selection
    (npop, dim) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, dim))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, 2)
        if rank[ind1] < rank[ind2]:
            mating_pool[i] = pop[ind1]
        elif rank[ind1] == rank[ind2]:
            mating_pool[i] = pop[ind1] if cd[ind1] > cd[ind2] else pop[ind2]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < pm / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def nd_cd_sort(pop, objs, rank, cd, npop):
    # sort the population according to the Pareto rank and crowding distance
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pop = np.zeros((npop, pop.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    next_rank = np.zeros(npop)
    next_cd = np.zeros(npop)
    for i in range(npop):
        next_pop[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
        next_rank[i] = temp_list[i][2]
        next_cd[i] = temp_list[i][3]
    return next_pop, next_objs, next_rank, next_cd


def main(pop, iter):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: upper bound
    :param ub: lower bound
    :param pc: crossover probability
    :param eta_c: spread factor distribution index
    :param pm: mutation probability
    :param eta_m: perturbance factor distribution index
    :return:
    """
    # Step 1. Initialization
    lb = np.array([0] * len(pop[0][0]))
    ub = np.array([1] * len(pop[0][0]))
    pc=1
    eta_c=20
    pm=0.1
    eta_m=20
    npop = len(pop)
    #dim = len(pop)  # dimension
    #pop = np.random.rand(npop, dim) * (ub - lb) + lb  # population
    objs = np.array([fc.cal_obj(pop[i]) for i in range(npop)])  # objectives
    #objs = fc.min_max_normalize(objs)
    [pfs, rank] = nd_sort(objs)  # Pareto rank
    cd = crowding_distance(objs, pfs)  # crowding distance
    lot_pool = np.zeros((0,len(pop[0][0])))
    fac_pool = np.zeros((0,len(pop[0][0])))
    for i in range(len(pop)):
        mid_lot = pop[i][0]
        mid_fac = pop[i][1]
        lot_pool = np.concatenate((lot_pool, mid_lot[np.newaxis,:]), axis=0)
        fac_pool = np.concatenate((fac_pool, mid_fac[np.newaxis,:]), axis=0)

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 2 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Selection + crossover + mutation
        mating_pool = selection(lot_pool, rank, cd, pc)
        offspring = crossover(mating_pool, lb, ub, eta_c)
        offspring = mutation(offspring, lb, ub, pm, eta_m)
        o_pop = []
        for i in range(len(pop)):
            mid_pop =  np.zeros((0,len(pop[0][0])))
            mid_pop = np.concatenate((mid_pop, offspring[i][np.newaxis,:]), axis=0)
            mid_pop = np.concatenate((mid_pop, fac_pool[i][np.newaxis,:]), axis=0)
            o_pop.append(mid_pop)
        new_objs = np.array([fc.cal_obj(o_pop[i]) for i in range(len(o_pop))])
        #new_objs = fc.min_max_normalize(new_objs)
        pop_semi = np.concatenate((lot_pool, offspring), axis=0)
        objs = np.concatenate((objs, new_objs), axis=0)
        [pfs, rank] = nd_sort(objs)
        cd = crowding_distance(objs, pfs)
        [lot_pool, objs, rank, cd] = nd_cd_sort(pop_semi, objs, rank, cd, npop)
    final_pop = []
    for i in range(len(lot_pool)):
        mid_pop =  np.zeros((0,len(pop[0][0])))
        mid_pop = np.concatenate((mid_pop, lot_pool[i][np.newaxis,:]), axis=0)
        mid_pop = np.concatenate((mid_pop, fac_pool[i][np.newaxis,:]), axis=0)
        final_pop.append(mid_pop)
    return o_pop,objs,rank

