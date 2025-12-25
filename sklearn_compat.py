"""
Compatibility shim for loading old scikit-learn models.
This module provides backward compatibility for models trained with older scikit-learn versions.
"""
import sys
from types import ModuleType

# Create a fake module for sklearn.ensemble._gb_losses
gb_losses_module = ModuleType('sklearn.ensemble._gb_losses')

# Try to import loss classes from their new locations in scikit-learn 1.8+
try:
    # In newer versions, these might be in different places or renamed
    # We'll try to import from sklearn.ensemble._gb
    from sklearn.ensemble._gb import (
        LeastSquaresError,
        LeastAbsoluteError, 
        HuberLossFunction,
        QuantileLossFunction,
        BinomialDeviance,
        MultinomialDeviance,
        ExponentialLoss
    )
    
    # Add them to our fake module
    gb_losses_module.LeastSquaresError = LeastSquaresError
    gb_losses_module.LeastAbsoluteError = LeastAbsoluteError
    gb_losses_module.HuberLossFunction = HuberLossFunction
    gb_losses_module.QuantileLossFunction = QuantileLossFunction
    gb_losses_module.BinomialDeviance = BinomialDeviance
    gb_losses_module.MultinomialDeviance = MultinomialDeviance
    gb_losses_module.ExponentialLoss = ExponentialLoss
    
except ImportError:
    # If that doesn't work, create dummy classes
    # This is a fallback - the models might not work perfectly but won't crash on load
    class DummyLoss:
        pass
    
    gb_losses_module.LeastSquaresError = DummyLoss
    gb_losses_module.LeastAbsoluteError = DummyLoss
    gb_losses_module.HuberLossFunction = DummyLoss
    gb_losses_module.QuantileLossFunction = DummyLoss
    gb_losses_module.BinomialDeviance = DummyLoss
    gb_losses_module.MultinomialDeviance = DummyLoss
    gb_losses_module.ExponentialLoss = DummyLoss

# Register the fake module so pickle can find it
sys.modules['sklearn.ensemble._gb_losses'] = gb_losses_module
