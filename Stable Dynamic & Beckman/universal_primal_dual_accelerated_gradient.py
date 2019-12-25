from math import sqrt
import numpy as np
from copy import deepcopy
import warnings


#criteria: stable dynamic 'dual_threshold' AND 'primal_threshold', 'dual_rel' AND 'primal_rel'. 

#beckman : + 'dual_gap_rel', 'dual_gap_threshold', 'primal_threshold', 'primal_rel'

#criteria: 'star_solution_residual',

#practice: 'dual_rel'

def binary_search(f, a, b, eps=1e-5, **kwargs):
    # TO DO
    phi = kwargs.get('phi')
    if phi is not None:
        grad_phi = phi.grad

    max_iter = kwargs.get('max_iter')
    if max_iter is None:
        warnings.warn("Max iter for binary search not specified")
        max_iter = 1000

    it = 0
    c = (a + b) / 2.0
    while abs(b - a) > eps and it < max_iter:
        # Check left subsegment
        y = (a + c) / 2.0
        if f(y) <= f(c):
            b = c
            c = y
        else:
            # Check right subsegment
            z = (b + c) / 2.0
            if f(c) <= f(z):
                a = y
                b = z
            else:
                a = c
                c = z
        it += 1
    print('Binary search results: abs(b-a)={}, it={}, beta={}'.format(abs(b - a), it, c))
    return c


def solve_quadratic_equation(a, b, c):
    D = b ** 2 - 4 * a * c
    if D < 0:
        raise ValueError("Couldn't solve quadratic equation. D < 0")
    elif D == 0:
        return -b / (2 * a),
    else:
        x1 = (-b + D ** 0.5) / (2 * a)
        x2 = (-b - D ** 0.5) / (2 * a)
        return x1, x2


def get_a_next(phi, grad, eta, lambda_, A, eps):
    delta_phi = (phi.func(eta) - phi.func(lambda_))
    ax = sum(grad ** 2)
    bx = 2 * delta_phi - eps
    cx = 2 * A * delta_phi
    roots = solve_quadratic_equation(ax, bx, cx)
    return roots[0]


def M(delta, nu, m_nu):
    return ((1 - nu) / (1 + nu) * m_nu / delta) ** (
                (1 - nu) / (1 + nu)) * m_nu


def universal_primal_dual_accelerated_gradient_function(
        phi_big_oracle,
        prox_h,
        primal_dual_oracle,
        t_start,
        max_iter=1000,
        crit_name='dual_gap_rel',
        eps=1e-5,
        eps_abs=None,
        verbose=False,
        H=100,
        total_od_flow=1e5
):
    if crit_name == 'dual_gap_rel':
        def crit():
            nonlocal duality_gap, duality_gap_init, eps
            return duality_gap < eps * duality_gap_init
    if crit_name == 'dual_rel':
        def crit():
            nonlocal dual_func_history, eps
            l = len(dual_func_history)
            return dual_func_history[l // 2] - dual_func_history[-1] \
                   < eps * (dual_func_history[0] - dual_func_history[-1])
    if crit_name == 'primal_rel':
        def crit():
            nonlocal primal_func_history, eps
            l = len(primal_func_history)
            return primal_func_history[l // 2] - primal_func_history[-1] \
                   < eps * (primal_func_history[0] - primal_func_history[-1])

    # for logging
    iter_step = 1
    dim = len(t_start)
    duality_gap_init = None

    primal_func_history = []
    dual_func_history = []
    duality_gap_history = []
        
    success = False

    M0 = 2 * np.sqrt(H) * total_od_flow

    a_prev = 0
    A_prev = a_prev
    grad_sum_prev = np.zeros(dim)
    lambda_ = np.random.rand(dim) / 10
    eta_prev = np.random.rand(dim) / 10
    zeta_prev  = np.random.rand(dim) / 10
    bin_search_kws = dict(max_iter=max_iter, eps=eps)
    for it_counter in range(max_iter):
        f = lambda b: phi_big_oracle.func(zeta_prev + b * (eta_prev - zeta_prev))
        beta = binary_search(f, 0, 1, **bin_search_kws)
        lambda_ = zeta_prev + beta * (eta_prev - zeta_prev)
        grad_phi = phi_big_oracle.grad(lambda_)
        grad_sum_next = grad_sum_prev + a_prev * grad_phi

        eps_arg = 1 if it_counter == 0 else a_prev / A_prev
        h = 1e-4# 1 / M(eps_arg * eps, 0, M0)
        eta_next = prox_h(lambda_ - h * grad_phi, 1/h, lambda_)
        a_next = get_a_next(phi_big_oracle, grad_phi, eta_next, lambda_, A_prev, eps)
        A_next = A_prev + a_next
        zeta_next = prox_h(zeta_prev - a_next * grad_phi, 1/a_next, zeta_prev)
        x = -grad_sum_next / A_next
        # print(h)

        zeta_prev = deepcopy(zeta_next)
        a_prev = deepcopy(a_next)
        A_prev = deepcopy(A_next)
        grad_sum_prev = deepcopy(grad_sum_next)
        eta_prev = deepcopy(eta_next)

        if it_counter == 1:
            duality_gap_init = primal_dual_oracle.duality_gap(eta_next, x)
            if eps_abs is None:
                eps_abs = eps * duality_gap_init

            if verbose:
                print('Primal_init = {:g}'.format(
                    primal_dual_oracle.primal_func_value(x)))
                print('Dual_init = {:g}'.format(
                    primal_dual_oracle.dual_func_value(eta_next)))
                print('Duality_gap_init = {:g}'.format(duality_gap_init))

        if it_counter > 0:
            primal_func_value = primal_dual_oracle.primal_func_value(x)
            dual_func_value = primal_dual_oracle.dual_func_value(eta_next)
            duality_gap = primal_dual_oracle.duality_gap(eta_next, x)

            primal_func_history.append(primal_func_value)
            dual_func_history.append(dual_func_value)
            duality_gap_history.append(duality_gap)

            if verbose and (it_counter == 1 or it_counter % iter_step == 0):
                print('\nIterations number: ' + str(it_counter))
                print('Primal_func_value = {:g}'.format(primal_func_value))
                print('Dual_func_value = {:g}'.format(dual_func_value))
                print('Duality_gap = {:g}'.format(duality_gap))
                print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init),
                      flush=True)

            if duality_gap < eps_abs:
               success = True
               break

    result = {'times': eta_next,
              'flows': x,
              'iter_num': it_counter,
              'duality_gap_history': duality_gap_history,
              'primal_func_history': primal_func_history,
              'dual_func_history': dual_func_history,
              }

    if success:
        result['res_msg'] = 'success'
    else:
        result['res_msg'] = 'iterations number exceeded'

    if verbose:
        if success:
            print('\nSuccess! Iterations number: ' + str(it_counter))
        else:
            print('\nIterations number exceeded!')
        print('Primal_func_value = {:g}'.format(primal_func_value))
        print('Duality_gap / Duality_gap_init = {:g}'.format(
            duality_gap / duality_gap_init))
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(
            phi_big_oracle.time))



    # /////////////////////////////////////
    
    # for it_counter in range(1,max_iter+1):
    #     inner_iters_num = 1
    #     while True:
    #         alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_prev / L_value)
    #         A = A_prev + alpha
    #
    #         y = (alpha * u_prev + A_prev * t_prev) / A
    #         phi_grad_y = phi_big_oracle.grad(y)
    #         grad_sum = grad_sum_prev + alpha * phi_grad_y
    #         u = prox_h(y_start - grad_sum, A, u_start = u_prev)
    #         t = (alpha * u + A_prev * t_prev) / A
    #
    #         if it_counter == 1 and inner_iters_num == 1:
    #             flows_weighted = - grad_sum / A
    #             duality_gap_init = primal_dual_oracle.duality_gap(t, flows_weighted)
    #             if eps_abs is None:
    #                 eps_abs = eps * duality_gap_init
    #
    #             if verbose:
    #                 print('Primal_init = {:g}'.format(primal_dual_oracle.primal_func_value(flows_weighted)))
    #                 print('Dual_init = {:g}'.format(primal_dual_oracle.dual_func_value(t)))
    #                 print('Duality_gap_init = {:g}'.format(duality_gap_init))
    #
    #         left_value = (phi_big_oracle.func(y) + np.dot(phi_grad_y, t - y) +
    #                       0.5 * alpha / A * eps_abs) - phi_big_oracle.func(t)
    #         right_value = - 0.5 * L_value * np.sum((t - y)**2)
    #         if left_value >= right_value:
    #             break
    #         else:
    #             L_value *= 2
    #             inner_iters_num += 1
    #
    #
    #     A_prev = A
    #     L_value /= 2
    #
    #     t_prev = t
    #     u_prev = u
    #     grad_sum_prev = grad_sum
    #     flows_weighted = - grad_sum / A
    #
    #     primal_func_value = primal_dual_oracle.primal_func_value(flows_weighted)
    #     dual_func_value = primal_dual_oracle.dual_func_value(t)
    #     duality_gap = primal_dual_oracle.duality_gap(t, flows_weighted)
    #
    #     primal_func_history.append(primal_func_value)
    #     dual_func_history.append(dual_func_value)
    #     duality_gap_history.append(duality_gap)
    #     inner_iters_history.append(inner_iters_num)
    #
    #     #if duality_gap < eps_abs:
    #     #    success = True
    #     #    break
    #
    #     if verbose and (it_counter == 1 or it_counter % iter_step == 0):
    #         print('\nIterations number: ' + str(it_counter))
    #         print('Inner iterations number: ' + str(inner_iters_num))
    #         print('Primal_func_value = {:g}'.format(primal_func_value))
    #         print('Dual_func_value = {:g}'.format(dual_func_value))
    #         print('Duality_gap = {:g}'.format(duality_gap))
    #         print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init), flush=True)
    #
    #
    # result = {'times': t,
    #           'flows': flows_weighted,
    #           'iter_num': it_counter,
    #           'duality_gap_history': duality_gap_history,
    #           'inner_iters_history': inner_iters_history,
    #           'primal_func_history': primal_func_history,
    #           'dual_func_history': dual_func_history,
    #          }
    #
    # if success:
    #     result['res_msg'] = 'success'
    # else:
    #     result['res_msg'] = 'iterations number exceeded'
    #
    # if verbose:
    #     if success:
    #         print('\nSuccess! Iterations number: ' + str(it_counter))
    #     else:
    #         print('\nIterations number exceeded!')
    #     print('Primal_func_value = {:g}'.format(primal_func_value))
    #     print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init))
    #     print('Phi_big_oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
    #     print('Inner iterations total number: ' + str(sum(inner_iters_history)))
    
    return result