import time
from Market import Market
from Option import Option
from PricingModel import ModelMC

market = Market(S0=1, r=0.06, sigma=0.2, dividend=0)
option = Option(K=1.1, T=3, opt_type="put")
model_mc = ModelMC(market, option, n_paths=8, n_steps=3, seed=1)

print("Prix Black-Scholes :", f"{option.black_scholes_price(market):.6f}")

start_total = time.time()

start = time.time()
price = model_mc.european_price_scalar()
end = time.time()
elapsed = end - start
total_elapsed = end - start_total
print(f"Prix Européen Scalaire: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

start = time.time()
price = model_mc.european_price_vectorized()
end = time.time()
elapsed = end - start
total_elapsed = end - start_total
print(f"Prix Européen Vectorisé: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

start = time.time()
price = model_mc.american_price_scalar()
end = time.time()
elapsed = end - start
total_elapsed = end - start_total
print(f"Prix Américain Scalaire: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

start = time.time()
price = model_mc.american_price_vectorized()
end = time.time()
elapsed = end - start
total_elapsed = end - start_total
print(f"Prix Américain Vectorisé: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

print(f"Temps total d'exécution: {time.time() - start_total:.4f} s")
