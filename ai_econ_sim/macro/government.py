"""Government tax revenue module (passive collector in v0)."""

from ai_econ_sim.macro.accounting import MacroAccounts
from ai_econ_sim.config import CORPORATE_TAX_RATE, PAYROLL_TAX_RATE, INCOME_TAX_RATE


def compute_tax_revenue(accts: MacroAccounts) -> float:
    """Re-derive tax revenue from macro accounts (for verification/reporting)."""
    payroll = accts.total_labor_income * PAYROLL_TAX_RATE
    income = accts.total_labor_income * INCOME_TAX_RATE
    corporate = accts.total_capital_income * CORPORATE_TAX_RATE
    return payroll + income + corporate
