#include "modelica/matching/Flow.hpp"

#include <fcntl.h>

#include "llvm/ADT/iterator_range.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/utils/IndexSet.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

void FlowCandidates::dump(llvm::raw_ostream& OS) const
{
	OS << "Flows (first one is the best one):\n";
	for (const auto& c : make_range(rbegin(choises), rend(choises))) {
		c.dump(OS);
  }
}

string FlowCandidates::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}

void Flow::dump(llvm::raw_ostream& OS) const
{
	edge->dump(OS);
	OS << "\t Forward=" << static_cast<int>(isForwardEdge()) << " Source Set:";
	set.dump(OS);
	OS << "-> Arriving Set:";
	mappedFlow.dump(OS);
	OS << "\n";
}

FlowCandidates::FlowCandidates(SmallVector<Flow, 2> c, const MatchingGraph& g)
		: choises(std::move(c))
{
	assert(find_if(choises.begin(), choises.end(), [](const Flow& flow) {
					 return flow.empty();
				 }) == choises.end());
	sort(choises, [&](const auto& l, const auto& r) {
		return Flow::compare(l, r, g);
	});
}

/* restituisce true se c'è ancora roba da matchare in questo path */
bool AugmentingPath::valid() const
{
	if (frontier.empty())
		return false;
	if (getCurrentCandidates().empty())
		return false;
  
  /* frontier alterna FlowCandidates avanti e flowcandidates indietro
   * Controllando che frontier sia dispari verifichiamo che finisca con un FlowCandidates
   * avanti e non uno indietro.
   * Se non è dispari vado da qualche parte, altrimenti nov*/
	if ((frontier.size() % 2) != 1)
		return false;

  /* se siamo qui, sicuramente l'ultimo FlowCandidates sarà fatto di archi
   * positivi (a causa del check pari/dispari) */
   
  /* Variable corrispondente all'ultimo arco nel FlowCandidates */
	const auto& currentVar = getCurrentCandidates().getCurrentVariable();
  /* getCurrentCandidates().getCurrent() == getCurrentFlow() */
  // ottieni l'IndexSet della variabile
	auto set = getCurrentFlow().getMappedSet();
  /* togli gli indici già matchati */
	set.remove(graph.getMatchedSet(currentVar));

	return !set.empty();
}

/* trova l'insieme di candidati "ovvi", cioè quelli forward che si trovano
 * immediatamente a partire dalle equazioni e dal **flusso non matchato** */
FlowCandidates AugmentingPath::selectStartingEdge() const
{
	SmallVector<Flow, 2> possibleStarts;

	for (const auto& eq : graph.getModel())
	{
		IndexSet eqUnmatched = graph.getUnmatchedSet(eq);
		if (eqUnmatched.empty())
			continue;

		for (Edge& e : graph.arcsOf(eq))
			possibleStarts.emplace_back(Flow::forwardedge(e, eqUnmatched));
	}

	return FlowCandidates(possibleStarts, graph);
}
/* (perché questo non è un metodo d'istanza?) */
static IndexSet possibleForwardFlow(
		const Flow& backEdge, const Edge& forwadEdge, const MatchingGraph& graph)
{
	assert(!backEdge.isForwardEdge());
	auto direct = backEdge.getSet();
	direct.intersecate(forwadEdge.getEquation().getInductions());
	return direct;
}

/* Crea la lista di archi che bisogna aggiungere al matching per far andare avanti
 * il flusso assumendo che venga smatchato l'arco indietro getCurrentFlow()
 * (attenzione: fa uno step solo! potrebbe restituire l'insieme vuoto) */
FlowCandidates AugmentingPath::getForwardMatchable() const
{
	assert(!getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> directMatch;

  /* calcola l'insieme di archi uscenti dall'equazione smatchata */
	auto connectedEdges = graph.arcsOf(getCurrentFlow().getEquation());
	for (Edge& edge : connectedEdges)
	{
    /* evita loop infiniti */
		if (&edge == &getCurrentFlow().getEdge())
			continue;
    /* calcola gli indici dove si può far passare il flusso */
		auto possibleFlow = possibleForwardFlow(getCurrentFlow(), edge, graph);
		if (!possibleFlow.empty())
      /* consideralo se ce ne sono */
			directMatch.emplace_back(Flow::forwardedge(edge, move(possibleFlow)));
	}

	return FlowCandidates(directMatch, graph);
}

IndexSet AugmentingPath::possibleBackwardFlow(const Edge& backEdge) const
{
	const Flow& forwardEdge = getCurrentFlow();
	assert(forwardEdge.isForwardEdge());
	auto alreadyAssigned = backEdge.map(backEdge.getSet()); /* indici da togliere lato variabile */
	auto possibleFlow = forwardEdge.getMappedSet(); /* indici assegnati lato variabile */
	alreadyAssigned.intersecate(possibleFlow);

  /* cerca se lo stesso arco è già stato smatchato e togli quello smatching da
   * questo (questo ciclo rende l'algoritmo di ricerca del path quadratico) */
	for (const auto& siblingSet : frontier)
	{
		const auto currentEdge = siblingSet.getCurrent();
		if (&currentEdge.getEdge() != &backEdge)
			continue;
		if (currentEdge.isForwardEdge())
			continue;

		alreadyAssigned.remove(currentEdge.getMappedSet());
	}

	return alreadyAssigned;
}

/* Crea la lista di archi che bisogna rimuovere dal matching assumendo che
 * venga scelto l'arco in avanti getCurrentFlow() (attenzione: fa uno step solo!) */
FlowCandidates AugmentingPath::getBackwardMatchable() const
{
	assert(getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> undoingMatch;

  /* archi connessi alla variabile "matchata" */
	auto connectedEdges = graph.arcsOf(getCurrentFlow().getVariable());
	for (Edge& edge : connectedEdges)
	{
    /* ignora se si tratta dell'arco entrante che stiamo considerando
     * (altrimenti entriamo in un ciclo infinito di matching/smatching dello
     * stesso arco) */
		if (&edge == &getCurrentFlow().getEdge())
			continue;
    /* calcola il flow che si può smatchare */
		auto backFlow = possibleBackwardFlow(edge);
		if (!backFlow.empty())
      /* se c'è qualcosa da smatchare, consideralo */
			undoingMatch.emplace_back(Flow::backedge(edge, move(backFlow)));
	}

	return FlowCandidates(undoingMatch, graph);
}

/* calcola il prossimo insieme di percorsi matchabili/smatchabili considerando
 * che l'obiettivo è smatchare/matchare l'arco restituito da getCurrentFlow */
FlowCandidates AugmentingPath::getBestCandidate() const
{
	if (!getCurrentFlow().isForwardEdge())
		return getForwardMatchable();

	return getBackwardMatchable();
}

AugmentingPath::AugmentingPath(MatchingGraph& graph, size_t maxDepth)
		: graph(graph), frontier({ /* cerca cose ovvie */ selectStartingEdge() })
{
  /* se siamo qui e selectStartingEdge() non ha trovato nulla che si possa
   * matchare, valid() sarà false ed entriamo nel ciclo successivo che
   * calcola il grafo residuale vero e proprio
   *
   * Se il matching non è completo e il sistema è propriamente specificato,
   * qualcosa da matchare in frontier ci sarà sicuramente, ma probabilmente
   * non è possibile matcharlo senza smatchare prima qualcos'altro */
  if (!valid()) {
    dbgs() << "*** MATCHING REQUIRES DFS\n";
    dbgs() << "initial frontier:\n";
    dump(errs());
  }
   
	while (!valid() && frontier.size() < maxDepth)
	{
		// in massimese: while the current siblings are not empty keep exploring
    // in italiano: abbiamo scelto un arco da dove partire, fai la DFS per
    //  capire cosa smatchare per matcharlo
    // getCurrentCandidates == frontier.back == un'istanza di FlowCandidates
		if (!getCurrentCandidates().empty() /* BUG QUANDO frontier È VUOTO??? */)
		{
      /* questo è un passo della DFS! */
			frontier.push_back(getBestCandidate());
      dbgs() << "*** DFS STEP\n";
      dump(errs());
			continue;
		}
  
    dbgs() << "*** BACKTRACKING\n";
    /* getCurrentCandidates().empty() == true e valid == false
     * la DFS non è riuscita a trovare un flusso positivo
     * Facciamo backtracking in modo da riprovare alla prossima iterazione */

		// if they are empty remove the last siblings group
    /* rimuovi il FlowCandidates vuoto aggiunto dall'ultimo passo fallito della DFS */
		frontier.erase(frontier.end() - 1);

		// if the frontier is now empty we are done
		// there is no good path
    /* AKA non ci sono più percorsi non provati */
		if (frontier.empty())
			return;

		// else remove one of the siblings
    /* elimina l'ultimo matching/smatching preso dalla DFS in modo da ricominciare
     * dal successivo in ordine di priorità */
		getCurrentCandidates().pop();
    
    dbgs() << "backtracked frontier:\n";
    dump(errs());
	}
}

void AugmentingPath::apply()
{
	assert(valid());

  // IndexSet degli indici matchati nella prima variabile da considerare
	auto alreadyMatchedVars = graph.getMatchedSet(getCurrentFlow().getVariable());
  /* set: IndexSet degli indici della variabile che È POSSIBILE matchare con
   * l'arco che stiamo considerando */
	auto set = getCurrentFlow().getMappedSet();
	set.remove(alreadyMatchedVars);
  /* set adesso è il flusso da applicare, partendo dalla fine del percorso
   * aumentante */

  /* fai il giochino di andare avanti e indietro nel grafo per applicare il
   * flusso e nel contempo cancellarlo nei punti dove l'abbiamo assorbito */
	auto reverseRange = make_range(rbegin(frontier), rend(frontier));
	for (auto& edge : reverseRange)
	{
    /* ottieni l'arco da matchare */
		Flow& flow = edge.getCurrent();
    /* aggiungi l'arco al matching e converti il flusso da variabile a
     * equazione e viceversa */
		set = flow.applyAndInvert(set);
	}
}

void AugmentingPath::dump(llvm::raw_ostream& OS) const
{
	OS << "valid path = " << (valid() ? "true" : "false") << '\n';
	OS << "frontier (last item is the current one):\n";
	for (const auto& e : frontier) {
    OS << "****** SET " << &e << " ******\n";
		e.dump(OS);
  }
}
string AugmentingPath::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}
void AugmentingPath::dumpGraph(
		raw_ostream& OS,
		bool displayEmptyEdges,
		bool displayMappings,
		bool displayOnlyMatchedCount,
		bool displayOtherOptions) const
{
	graph.dumpGraph(
			OS, displayEmptyEdges, displayMappings, displayOnlyMatchedCount, false);

	size_t candidateCount = 0;
	for (const auto& candidate : frontier)
	{
		size_t edgeIndex = 0;
		for (const auto& edge : candidate)
		{
			if (!displayOtherOptions && &edge != &candidate.getCurrent())
				continue;

			if (edge.isForwardEdge())
			{
				OS << "Eq_" << graph.indexOfEquation(edge.getEquation());
				OS << " -> " << edge.getVariable().getName();
			}
			else
			{
				OS << edge.getVariable().getName() << "->";
				OS << "Eq_" << graph.indexOfEquation(edge.getEquation());
			}
			OS << " [color=";
			OS << (&edge == &candidate.getCurrent() ? "gold" : "green");
			OS << ", label=" << candidateCount;
			OS << "];\n";
			edgeIndex++;
		}
		candidateCount++;
	}

	OS << "}\n";
}
