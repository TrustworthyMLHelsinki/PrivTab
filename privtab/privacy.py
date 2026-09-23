"""GDP conversion and replace-one DP-SGD accounting."""
import functools
import math
from typing import Optional

from dp_accounting.pld.privacy_loss_distribution import NeighborRel, from_gaussian_mechanism
from .gdp_converter import PLDConverter



def compute_mu_regret_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
):
    pld = from_gaussian_mechanism(
        standard_deviation=sigma,
        sensitivity=max_grad_norm,
        sampling_prob=q,
        use_connect_dots=True,
        value_discretization_interval=value_discretization_interval,
        neighboring_relation=neighboring_relation,
    ).self_compose(compositions)
    converter = PLDConverter(pld)
    try:
        mu, regret = converter.get_mu_and_regret(err=err)
    except ValueError as exc:
        if "zero-size array to reduction operation minimum" not in str(exc):
            raise
        return math.inf, math.inf
    return mu, regret


def compute_mu_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
):
    pld = from_gaussian_mechanism(
        standard_deviation=sigma,
        sensitivity=max_grad_norm,
        sampling_prob=q,
        use_connect_dots=True,
        value_discretization_interval=value_discretization_interval,
        neighboring_relation=neighboring_relation,
    ).self_compose(compositions)
    converter = PLDConverter(pld)
    try:
        return converter.get_mu(err=err)
    except ValueError as exc:
        if "zero-size array to reduction operation minimum" not in str(exc):
            raise
        return math.inf


@functools.lru_cache(maxsize=2048)
def _cached_mu_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
):
    return compute_mu_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
    )


@functools.lru_cache(maxsize=1024)
def _cached_mu_regret_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
):
    return compute_mu_regret_from_continuous_gaussian_pld_with_neighbor_relation(
        sigma,
        max_grad_norm,
        q,
        compositions,
        value_discretization_interval,
        err,
        neighboring_relation,
    )


def find_dpsgd_sigma_bracket(
        target_mu: float,
        max_grad_norm: float,
        q: float,
        compositions: int,
        neighboring_relation,
        sigma_low: Optional[float] = None,
        sigma_high: Optional[float] = None,
        value_discretization_interval=1e-4,
        mu_conversion_tol: float = 1e-6,
        max_iterations=100,
        mu_gdp_sigma_scale: float = 2.0,
):
    mu_gdp_sigma = mu_gdp_sigma_scale * max_grad_norm * math.sqrt(compositions) / target_mu
    if sigma_high is None:
        sigma_high = mu_gdp_sigma * q * 2
    if sigma_low is None:
        sigma_low = max(mu_gdp_sigma * q / 2, 1e-12)

    def eval_mu(sigma: float) -> float:
        return _cached_mu_from_continuous_gaussian_pld_with_neighbor_relation(
            sigma,
            max_grad_norm,
            q,
            compositions,
            value_discretization_interval,
            mu_conversion_tol,
            neighboring_relation,
        )

    mu_high = eval_mu(sigma_high)
    mu_low = eval_mu(sigma_low)

    bracket_expansions = 0
    while mu_high > target_mu and bracket_expansions < max_iterations:
        sigma_high *= 2
        mu_high = eval_mu(sigma_high)
        bracket_expansions += 1
    if mu_high > target_mu:
        raise ValueError("sigma_high produced a mu that is larger than target_mu.")

    while mu_low < target_mu and bracket_expansions < 2 * max_iterations:
        sigma_low /= 2
        if sigma_low <= 0:
            break
        mu_low = eval_mu(sigma_low)
        bracket_expansions += 1
    if mu_low < target_mu:
        raise ValueError("sigma_low produced a mu that is smaller than target_mu.")

    return sigma_low, sigma_high


def compute_dpsgd_sigma_for_substitute_dp_fast(
        target_mu: float,
        max_grad_norm: float,
        q: float,
        compositions: int,
        sigma_low: Optional[float] = None,
        sigma_high: Optional[float] = None,
        value_discretization_interval=1e-4,
        mu_conversion_tol: float = 1e-6,
        sigma_search_tol: float = 1e-6,
        max_iterations=100,
):
    return compute_dpsgd_sigma_fast(
        target_mu=target_mu,
        max_grad_norm=max_grad_norm,
        q=q,
        compositions=compositions,
        neighboring_relation=NeighborRel.REPLACE_ONE,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        value_discretization_interval=value_discretization_interval,
        mu_conversion_tol=mu_conversion_tol,
        sigma_search_tol=sigma_search_tol,
        max_iterations=max_iterations,
        mu_gdp_sigma_scale=2.0,
    )


def compute_dpsgd_sigma_fast(
        target_mu: float,
        max_grad_norm: float,
        q: float,
        compositions: int,
        neighboring_relation,
        sigma_low: Optional[float] = None,
        sigma_high: Optional[float] = None,
        value_discretization_interval=1e-4,
        mu_conversion_tol: float = 1e-6,
        sigma_search_tol: float = 1e-6,
        max_iterations=100,
        mu_gdp_sigma_scale: float = 2.0,
):
    def eval_mu(sigma: float) -> float:
        return _cached_mu_from_continuous_gaussian_pld_with_neighbor_relation(
            sigma,
            max_grad_norm,
            q,
            compositions,
            value_discretization_interval,
            mu_conversion_tol,
            neighboring_relation,
        )

    def eval_mu_regret(sigma: float) -> tuple[float, float]:
        return _cached_mu_regret_from_continuous_gaussian_pld_with_neighbor_relation(
            sigma,
            max_grad_norm,
            q,
            compositions,
            value_discretization_interval,
            mu_conversion_tol,
            neighboring_relation,
        )

    sigma_low, sigma_high = find_dpsgd_sigma_bracket(
        target_mu=target_mu,
        max_grad_norm=max_grad_norm,
        q=q,
        compositions=compositions,
        neighboring_relation=neighboring_relation,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        value_discretization_interval=value_discretization_interval,
        mu_conversion_tol=mu_conversion_tol,
        max_iterations=max_iterations,
        mu_gdp_sigma_scale=mu_gdp_sigma_scale,
    )
    mu_high = eval_mu(sigma_high)
    mu_low = eval_mu(sigma_low)

    log_sigma_low = math.log(sigma_low)
    log_sigma_high = math.log(sigma_high)
    sigma = math.exp((log_sigma_high + log_sigma_low) / 2)
    mu = eval_mu(sigma)

    i = 0
    last_sigma = sigma_high
    last_mu = mu_high

    while i < max_iterations:
        if target_mu - sigma_search_tol <= mu < target_mu:
            final_sigma = sigma
            final_mu, final_regret = eval_mu_regret(final_sigma)
            return final_sigma, final_mu, final_regret

        prev_sigma = sigma
        if mu < target_mu:
            log_sigma_high = math.log(sigma)
            mu_high = mu
            last_sigma = sigma
            last_mu = mu
        if mu > target_mu:
            log_sigma_low = math.log(sigma)
            mu_low = mu

        next_log_sigma = None
        if math.isfinite(mu_low) and math.isfinite(mu_high) and mu_low > mu_high:
            secant_denom = mu_high - mu_low
            if abs(secant_denom) > 1e-15:
                secant_weight = (target_mu - mu_low) / secant_denom
                secant_weight = min(max(secant_weight, 0.05), 0.95)
                next_log_sigma = log_sigma_low + secant_weight * (log_sigma_high - log_sigma_low)

        if next_log_sigma is None:
            next_log_sigma = (log_sigma_high + log_sigma_low) / 2

        sigma = math.exp(next_log_sigma)
        mu = eval_mu(sigma)

        if abs(prev_sigma - sigma) < 1e-12:
            final_sigma = last_sigma
            final_mu, final_regret = eval_mu_regret(final_sigma)
            return final_sigma, final_mu, final_regret

        i += 1

    raise Exception("The algorithm did not find a sigma within the given number of iterations.")
