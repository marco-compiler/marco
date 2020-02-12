model TransmissionLineEquations
  parameter Real L = 10"length of the transmission line";
  final parameter Real l = L / 3 "length of the each segment";
  parameter Real res = 4"resistance per meter";
  parameter Real cap = 2"capacitance per meter";
  parameter Real induttance  = 3"induttanceuctance per meter";
  final parameter Real RL = (induttance / cap) ^ (1 / 2)
	"load resistance";
  parameter Real w = 3;
  final parameter Real v = 1 / (induttance * cap) ^ (1 / 2)
	"velocity of the signal";
  final parameter Real TD = L / v
	"time delay of the transmission line";
  parameter Real Vstep = 1 "input step voltage";
  Real[3] cur(start = 0.0)
	"current values at the nodes of the transmission line";
  Real[3] vol(start = 0.0)
	"voltage values at the nodes of the transmission line";
  Real[1] vvol(start = 0.0)"derivative of input voltage";
equation
  vvol = der(vol[1]);
  Vstep = vol[1] + 2 * (1 / w) * der(vol[1]) + 1 / w * w * der(vvol);
  vol[3] = cur[3] * RL;
  for i in 1:2 loop
	cap * der(vol[i + 1]) = (cur[i] - cur[i + 1]) / l;
	induttance * der(cur[i]) = (-res * cur[i]) - (vol[i + 1] - vol[i]) / l;
  end for;
end TransmissionLineEquations;
