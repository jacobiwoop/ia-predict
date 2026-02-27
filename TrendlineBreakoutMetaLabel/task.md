# Meta-Labeling Modularization Project

## Objectives
- [ ] Refactor the codebase to separate Strategy Logic from AI Meta-Labeling Logic
- [ ] Create a `Strategy` interface/base class
- [ ] Create a new moving average crossover strategy to prove the concept
- [ ] Ensure the Machine Learning pipeline can train on any strategy output

## Plan
1. [ ] **Define the Interface**: Create a standard format that all strategies must return (Entry Price, Stop Loss, Take Profit, Time Limit, Technical Features).
2. [ ] **Refactor Base Code**: Modify `trendline_break_dataset.py` to become a generic `ml_dataset_generator.py` that takes a Strategy object.
3. [ ] **Adapt Old Strategy**: Wrap the trendline code into this new `Strategy` structure.
4. [ ] **Create New Strategy**: Implement `MovingAverageCrossStrategy` to test the modularity.
5. [ ] **Test Pipeline**: Run `walkforward.py` on the new strategy.
