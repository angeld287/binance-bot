# Liquidity Sweep strategy notes

- Cuando la primera orden (long/short) se ejecuta y la opuesta se cancela, el Take Profit (TP) se calcula con prioridad en el precio de esa orden opuesta cancelada. Ese valor actúa como resistencia inmediata en longs o soporte inmediato en shorts.
- Si no se dispone de ese precio, los longs buscan la resistencia válida más cercana por encima del precio de entrada y los shorts el soporte más cercano por debajo. Los niveles se recalculan con `compute_levels` y se aplican respetando el `tickSize` de la bolsa.
