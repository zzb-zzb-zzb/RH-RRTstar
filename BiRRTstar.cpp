#include "BiRRTstar.h"
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <limits>
#include <vector>
#include "ompl/base/Goal.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/goals/GoalState.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/samplers/InformedStateSampler.h"
#include "ompl/base/samplers/informed/RejectionInfSampler.h"
#include "ompl/base/samplers/informed/OrderedInfSampler.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/util/GeometricEquations.h"

ompl::geometric::BiRRTstar::BiRRTstar(const base::SpaceInformationPtr &si)
  : base::Planner(si, "BiRRTstar"),
    space_(std::make_shared<base::RealVectorStateSpace>(2)), 
    ss_(std::make_shared<geometric::SimpleSetup>(space_))
{
    specs_.approximateSolutions = true;
    specs_.optimizingPaths = true;
    specs_.canReportIntermediateSolutions = true;

    Planner::declareParam<double>("range", this, &BiRRTstar::setRange, &BiRRTstar::getRange, "0.:1.:10000.");
    Planner::declareParam<double>("goal_bias", this, &BiRRTstar::setGoalBias, &BiRRTstar::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("rewire_factor", this, &BiRRTstar::setRewireFactor, &BiRRTstar::getRewireFactor,
                                  "1.0:0.01:2.0");
    Planner::declareParam<bool>("use_k_nearest", this, &BiRRTstar::setKNearest, &BiRRTstar::getKNearest, "0,1");
    Planner::declareParam<bool>("delay_collision_checking", this, &BiRRTstar::setDelayCC, &BiRRTstar::getDelayCC, "0,1");
    Planner::declareParam<bool>("tree_pruning", this, &BiRRTstar::setTreePruning, &BiRRTstar::getTreePruning, "0,1");
    Planner::declareParam<double>("prune_threshold", this, &BiRRTstar::setPruneThreshold, &BiRRTstar::getPruneThreshold,
                                  "0.:.01:1.");
    Planner::declareParam<bool>("pruned_measure", this, &BiRRTstar::setPrunedMeasure, &BiRRTstar::getPrunedMeasure, "0,1");
    Planner::declareParam<bool>("informed_sampling", this, &BiRRTstar::setInformedSampling, &BiRRTstar::getInformedSampling,
                                "0,1");
    Planner::declareParam<bool>("sample_rejection", this, &BiRRTstar::setSampleRejection, &BiRRTstar::getSampleRejection,
                                "0,1");
    Planner::declareParam<bool>("new_state_rejection", this, &BiRRTstar::setNewStateRejection,
                                &BiRRTstar::getNewStateRejection, "0,1");
    Planner::declareParam<bool>("use_admissible_heuristic", this, &BiRRTstar::setAdmissibleCostToCome,
                                &BiRRTstar::getAdmissibleCostToCome, "0,1");
    Planner::declareParam<bool>("ordered_sampling", this, &BiRRTstar::setOrderedSampling, &BiRRTstar::getOrderedSampling,
                                "0,1");
    Planner::declareParam<unsigned int>("ordering_batch_size", this, &BiRRTstar::setBatchSize, &BiRRTstar::getBatchSize,
                                        "1:100:1000000");
    Planner::declareParam<bool>("focus_search", this, &BiRRTstar::setFocusSearch, &BiRRTstar::getFocusSearch, "0,1");
    Planner::declareParam<unsigned int>("number_sampling_attempts", this, &BiRRTstar::setNumSamplingAttempts,
                                        &BiRRTstar::getNumSamplingAttempts, "10:10:100000");

    addPlannerProgressProperty("iterations INTEGER", [this] { return numIterationsProperty(); });
    addPlannerProgressProperty("best cost REAL", [this] { return bestCostProperty(); });
    auto bounds = dynamic_cast<base::SE2StateSpace*>(si_->getStateSpace().get())->getBounds();
    space_->as<base::RealVectorStateSpace>()->setBounds(bounds);
    rev_si_ = ss_->getSpaceInformation();
    rev_si_->setStateValidityChecker(([this](const base::State* state) { return collision_checker_->rev_isValid(state); }));
    rev_si_->setStateValidityCheckingResolution(0.001);
    rev_stateSpace_ = rev_si_->getStateSpace().get();
}

ompl::geometric::BiRRTstar::~BiRRTstar()
{
    freeMemory();
    rev_freeMemory();
}

void ompl::geometric::BiRRTstar::setup()
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);
    if (!si_->getStateSpace()->hasSymmetricDistance() || !si_->getStateSpace()->hasSymmetricInterpolate())
    {
        OMPL_WARN("%s requires a state space with symmetric distance and symmetric interpolation.", getName().c_str());
    }

    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });

    if (!rev_nn_)
        rev_nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    rev_nn_->setDistanceFunction([this](const Motion *a, const Motion *b) { return rev_distanceFunction(a, b); });
    // Setup optimization objective
    //
    // If no optimization objective was specified, then default to
    // optimizing path length as computed by the distance() function
    // in the state space.
    if (pdef_)
    {
        if (pdef_->hasOptimizationObjective())
            opt_ = pdef_->getOptimizationObjective();
        else
        {
            OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length for the allowed "
                        "planning time.",
                        getName().c_str());
            opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);

            // Store the new objective in the problem def'n
            pdef_->setOptimizationObjective(opt_);
        }

        // Set the bestCost_ and prunedCost_ as infinite
        bestCost_ = opt_->infiniteCost();
        prunedCost_ = opt_->infiniteCost();
        rev_opt_ = std::make_shared<base::PathLengthOptimizationObjective>(rev_si_);
    }
    else
    {
        OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
        setup_ = false;
    }

    // Get the measure of the entire space:
    prunedMeasure_ = si_->getSpaceMeasure();

    // Calculate some constants:
    calculateRewiringLowerBounds();
}

void ompl::geometric::BiRRTstar::clear()
{
    setup_ = false;
    Planner::clear();
    sampler_.reset();
    infSampler_.reset();
    freeMemory();
    rev_freeMemory();

    if (nn_)
        nn_->clear();
    if (rev_nn_)
        rev_nn_->clear();

    bestGoalMotion_ = nullptr;
    rev_bestGoalMotion_ = nullptr;
    goalMotions_.clear();
    rev_goalMotions_.clear();
    startMotions_.clear();

    iterations_ = 0;
    bestCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
    rev_bestCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
    prunedCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
    prunedMeasure_ = 0.0;
    heuristicBias_=init_heuristicBias_;
}

ompl::base::PlannerStatus ompl::geometric::BiRRTstar::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);
    auto *goal_state = dynamic_cast<base::GoalState *>(goal);

    base::State *start_state = si_->allocState();

    bool symCost = opt_->isSymmetric();

    // Check if there are more starts
    if (pis_.haveMoreStartStates() == true)
    {
        // There are, add them
        while (const base::State *st = pis_.nextStart())
        {
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, st);
            si_->copyState(start_state, st);
            motion->cost = opt_->identityCost();
            nn_->add(motion);
            startMotions_.push_back(motion);
        }

        // And assure that, if we're using an informed sampler, it's reset
        infSampler_.reset();
    }

    base::State *rev_st;
    rev_st=goal_state->getState();
    base::State *rev_start =rev_si_->allocState();
    rev_start->as<base::RealVectorStateSpace::StateType>()->values[0] = rev_st->as<base::SE2StateSpace::StateType>()->getX();
    rev_start->as<base::RealVectorStateSpace::StateType>()->values[1] = rev_st->as<base::SE2StateSpace::StateType>()->getY();
    auto *rev_motion = new Motion(rev_si_);
    rev_si_->copyState(rev_motion->state, rev_start);
    // OMPL_INFORM("%f,%f",rev_st->as<base::SE2StateSpace::StateType>()->getX(),rev_st->as<base::SE2StateSpace::StateType>()->getY());
    rev_nn_->add(rev_motion);

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    // Allocate a sampler if necessary
    if (!sampler_ && !infSampler_)
    {
        allocSampler();
    }
    if (!rev_sampler_)
        rev_sampler_ = rev_si_->allocStateSampler();

    OMPL_INFORM("%s: Started planning with %u states. Seeking a solution better than %.5f.", getName().c_str(), nn_->size(), opt_->getCostThreshold().value());

    if ((useTreePruning_ || useRejectionSampling_ || useInformedSampling_ || useNewStateRejection_) &&
        !si_->getStateSpace()->isMetricSpace())
        OMPL_WARN("%s: The state space (%s) is not metric and as a result the optimization objective may not satisfy "
                  "the triangle inequality. "
                  "You may need to disable pruning or rejection.",
                  getName().c_str(), si_->getStateSpace()->getName().c_str());

    const base::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

    Motion *approxGoalMotion = nullptr;
    double approxDist = std::numeric_limits<double>::infinity();

    auto *rmotion = new Motion(si_);
    base::State *rstate = rmotion->state;
    base::State *xstate = si_->allocState();

    base::State* temp=si_->allocState();

    Motion *rev_solution = nullptr;
    auto *rev_rmotion = new Motion(rev_si_);
    base::State *rev_rstate = rev_rmotion->state;
    base::State *rev_xstate = rev_si_->allocState();

    std::vector<Motion *> nbh;
    std::vector<Motion *> rev_nbh;

    std::vector<base::Cost> costs;
    std::vector<base::Cost> incCosts;
    std::vector<std::size_t> sortedCostIndices;

    std::vector<base::Cost> rev_costs;
    std::vector<base::Cost> rev_incCosts;
    std::vector<std::size_t> rev_sortedCostIndices;

    std::vector<int> valid;
    unsigned int rewireTest = 0;
    unsigned int statesGenerated = 0;

    std::vector<int> rev_valid;
    unsigned int rev_rewireTest = 0;
    unsigned int rev_statesGenerated = 0;

    if (bestGoalMotion_)
        OMPL_INFORM("%s: Starting planning with existing solution of cost %.5f", getName().c_str(),
                    bestCost_.value());

    if (useKNearest_)
        OMPL_INFORM("%s: Initial k-nearest value of %u", getName().c_str(),
                    (unsigned int)std::ceil(k_rrt_ * log((double)(nn_->size() + 1u))));
    else
        OMPL_INFORM(
            "%s: Initial rewiring radius of %.2f", getName().c_str(),
            std::min(maxDistance_, r_rrt_ * std::pow(log((double)(nn_->size() + 1u)) / ((double)(nn_->size() + 1u)),
                                                     1 / (double)(si_->getStateDimension()))));
    time::point start_time=time::now();

    // our functor for sorting nearest neighbors
    CostIndexCompare compareFn(costs, *opt_);
    CostIndexCompare rev_compareFn(rev_costs, *rev_opt_);

    base::State *rev_goal_state = rev_si_->allocState();
    rev_goal_state->as<base::RealVectorStateSpace::StateType>()->values[0] = start_state->as<base::SE2StateSpace::StateType>()->getX();
    rev_goal_state->as<base::RealVectorStateSpace::StateType>()->values[1] = start_state->as<base::SE2StateSpace::StateType>()->getY();
    // OMPL_INFORM("%f,%f",rev_goal_state->as<base::RealVectorStateSpace::StateType>()->values[0],rev_goal_state->as<base::RealVectorStateSpace::StateType>()->values[1]);


    bool rev = true;
    bool heuristic= false;
    bool use_heuristic = false;

    double rev_maxDistance_ = maxDistance_*rev_stepsize;
    OMPL_INFORM("maxDistance:%f",maxDistance_);

    while (ptc == false)
    {
        iterations_++;

        if (rev)
        {
            if(rng_.uniform01() < goalBias_ )
                rev_si_->copyState(rev_rstate, rev_goal_state);
            else
                rev_sampler_->sampleUniform(rev_rstate);
            /* find closest state in the tree */
            Motion *rev_nmotion = rev_nn_->nearest(rev_rmotion);
            base::State *rev_dstate = rev_rstate;

            /* find state to add */
            double rev_d = rev_si_->distance(rev_nmotion->state, rev_rstate);
            // OMPL_INFORM("rev_d:%f",rev_d);
            if (rev_d > rev_maxDistance_)
            {
                rev_si_->getStateSpace()->interpolate(rev_nmotion->state, rev_rstate, rev_maxDistance_/rev_d, rev_xstate);
                rev_dstate = rev_xstate;
            }
            if (revcheckMotion(rev_nmotion->state, rev_dstate))
            {
                auto *rev_motion = new Motion(rev_si_);
                rev_si_->copyState(rev_motion->state, rev_dstate);
                rev_motion->parent = rev_nmotion;
                
                rev_motion->incCost = rev_opt_->motionCost(rev_nmotion->state, rev_motion->state);
                rev_motion->cost = rev_opt_->combineCosts(rev_nmotion->cost, rev_motion->incCost);
                rev_getNeighbors(rev_motion, rev_nbh);
                rev_rewireTest += rev_nbh.size();
                ++rev_statesGenerated;

                if (rev_costs.size() < rev_nbh.size())
                {
                    rev_costs.resize(rev_nbh.size());
                    rev_incCosts.resize(rev_nbh.size());
                    rev_sortedCostIndices.resize(rev_nbh.size());
                }

                if (rev_valid.size() < rev_nbh.size())
                    rev_valid.resize(rev_nbh.size());
                std::fill(rev_valid.begin(), rev_valid.begin() + rev_nbh.size(), 0);

                // calculate all costs and distances
                for (std::size_t i = 0; i < rev_nbh.size(); ++i)
                {
                    rev_incCosts[i] = rev_opt_->motionCost(rev_nbh[i]->state, rev_motion->state);
                    rev_costs[i] = rev_opt_->combineCosts(rev_nbh[i]->cost, rev_incCosts[i]);
                }

                // sort the nodes
                for (std::size_t i = 0; i < rev_nbh.size(); ++i)
                    rev_sortedCostIndices[i] = i;
                std::sort(rev_sortedCostIndices.begin(), rev_sortedCostIndices.begin() + rev_nbh.size(), rev_compareFn);

                // collision check until a valid motion is found
                for (std::vector<std::size_t>::const_iterator i = rev_sortedCostIndices.begin();
                     i != rev_sortedCostIndices.begin() + rev_nbh.size(); ++i)
                {
                    if (rev_nbh[*i] == rev_nmotion ||
                        ((!useKNearest_ || rev_si_->distance(rev_nbh[*i]->state, rev_motion->state) < rev_maxDistance_) &&
                         revcheckMotion(rev_nbh[*i]->state, rev_motion->state)))
                    {
                        rev_motion->incCost = rev_incCosts[*i];
                        rev_motion->cost = rev_costs[*i];
                        rev_motion->parent = rev_nbh[*i];
                        rev_valid[*i] = 1;
                        break;
                    }
                    else
                        rev_valid[*i] = -1;
                }
                rev_nn_->add(rev_motion);
                rev_motion->parent->children.push_back(rev_motion);
                bool rev_checkForSolution = false;
                for (std::size_t i = 0; i < rev_nbh.size(); ++i)
                {
                    if (rev_nbh[i] != rev_motion->parent)
                    {
                        base::Cost rev_nbhIncCost;
                        rev_nbhIncCost = rev_incCosts[i];

                        base::Cost rev_nbhNewCost = rev_opt_->combineCosts(rev_motion->cost, rev_nbhIncCost);
                        if (rev_opt_->isCostBetterThan(rev_nbhNewCost, rev_nbh[i]->cost))
                        {
                            bool rev_motionValid;
                            if (rev_valid[i] == 0)
                            {
                                rev_motionValid =
                                    (!useKNearest_ || rev_si_->distance(rev_nbh[i]->state, rev_motion->state) < rev_maxDistance_) &&
                                    revcheckMotion(rev_motion->state, rev_nbh[i]->state);
                            }
                            else
                            {
                                rev_motionValid = (rev_valid[i] == 1);
                            }

                            if (rev_motionValid)
                            {
                                // Remove this node from its parent list
                                removeFromParent(rev_nbh[i]);

                                // Add this node to the new parent
                                rev_nbh[i]->parent = rev_motion;
                                rev_nbh[i]->incCost = rev_nbhIncCost;
                                rev_nbh[i]->cost = rev_nbhNewCost;
                                rev_nbh[i]->parent->children.push_back(rev_nbh[i]);

                                // Update the costs of the node's children
                                updateChildCosts(rev_nbh[i]);

                                rev_checkForSolution = true;
                            }
                        }
                    }
                }
                if (!heuristic && rev_nmotion->parent != nullptr)//rev_meet
                {
                    auto *cmotion = new Motion(si_);
                    convertRealVectorToSE2(rev_nmotion, cmotion->state, false);

                    double rcheck = si_->distance(cmotion->state, nn_->nearest(cmotion)->state);
                    // OMPL_INFORM("rcheck:%f");

                    if (rcheck < r_meet_ && si_->checkMotion(cmotion->state, nn_->nearest(cmotion)->state))
                    {
                        OMPL_INFORM("heuristic");
                        heuristic = true;
                        use_heuristic=true;
                        rev = false;
                        rev_solution = rev_nmotion;
                        while (rev_solution != nullptr)
                        {
                            heuristic_path.push_back(rev_solution);
                            rev_solution = rev_solution->parent;
                        }
                        double duration = time::seconds(time::now() - start_time);
                        OMPL_INFORM("Heuristic time: %f seconds", duration);
                        OMPL_INFORM("num:%ld", heuristic_path.size());
                        for (int i = 0; i < heuristic_path.size() - 1; ++i)
                        {
                            heuristic_idx.push_back(i);
                        }
                    }
                }
                if (rev_si_->equalStates(rev_motion->state, rev_goal_state))//rev satisfies the goal
                {
                    rev_goalMotions_.push_back(rev_motion);
                    rev_checkForSolution = true;
                }
                if (rev_checkForSolution)
                {
                    bool rev_updatedSolution = false;
                    if (!rev_bestGoalMotion_ && !rev_goalMotions_.empty())
                    {
                        // We have found our first solution, store it as the best. We only add one
                        // vertex at a time, so there can only be one goal vertex at this moment.
                        rev_bestGoalMotion_ = rev_goalMotions_.front();
                        rev_bestCost_ = rev_bestGoalMotion_->cost;
                        rev_updatedSolution = true;

                        OMPL_INFORM("Found an rev initial solution ");
                    }
                    else
                    {
                        // We already have a solution, iterate through the list of goal vertices
                        // and see if there's any improvement.
                        for (auto &rev_goalMotion : rev_goalMotions_)
                        {
                            // Is this goal motion better than the (current) best?
                            if (rev_opt_->isCostBetterThan(rev_goalMotion->cost, rev_bestCost_))
                            {
                                rev_bestGoalMotion_ = rev_goalMotion;
                                rev_bestCost_ = rev_bestGoalMotion_->cost;
                                rev_updatedSolution = true;

                                // Check if it satisfies the optimization objective, if it does, break the for loop
                                if (rev_opt_->isSatisfied(rev_bestCost_))
                                {
                                    break;
                                }
                            }
                        }
                    }

                    if (rev_updatedSolution)
                    {
                        if (useTreePruning_)
                        {
                            pruneTree(rev_bestCost_);
                        }

                        heuristic_path.clear();
                        heuristic_idx.clear();

                        Motion *rev_intermediate_solution =
                            rev_bestGoalMotion_->parent; // Do not include goal state to simplify code.

                        // Push back until we find the start, but not the start itself
                        while (rev_intermediate_solution->parent != nullptr)
                        {
                            heuristic_path.push_back(rev_intermediate_solution);
                            rev_intermediate_solution = rev_intermediate_solution->parent;
                        }
                        for (int i = 0; i < heuristic_path.size(); ++i)
                        {
                            heuristic_idx.push_back(i);
                        }
                        use_heuristic=true;//update_heuristic
                        heuristicBias_=heuristicBias_*alpha;
                    }
                }
            }
        }

        // sample random state (with goal biasing)
        int select_idx=0;
        int sequential_idx = 0;
        if (use_heuristic && !heuristic_idx.empty() && rng_.uniform01() < heuristicBias_)
        {
            // select_idx=sequential_idx%heuristic_idx.size();
            // Motion* selected = heuristic_path[heuristic_idx[select_idx]];
            select_idx=rng_.uniformInt(0, heuristic_idx.size()-1);
            Motion* selected = heuristic_path[heuristic_idx[select_idx]];
            convertRealVectorToSE2(selected, rstate, true);
            sequential_idx++;
        }
        else if (goal_s && goalMotions_.size() < goal_s->maxSampleCount() && rng_.uniform01() < goalBias_ &&
            goal_s->canSample())
            goal_s->sampleGoal(rstate);
        else
        {
            // Attempt to generate a sample, if we fail (e.g., too many rejection attempts), skip the remainder of this
            // loop and return to try again
            if (!sampleUniform(rstate))
                continue;
        }

        // find closest state in the tree
        Motion *nmotion = nn_->nearest(rmotion);

        if (intermediateSolutionCallback && si_->equalStates(nmotion->state, rstate))
            continue;

        base::State *dstate = rstate;

        // find state to add to the tree
        double d = si_->distance(nmotion->state, rstate);
        if (d > maxDistance_)
        {
            si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
            dstate = xstate;
        }

        // Check if the motion between the nearest state and the state to add is valid
        if (si_->checkMotion(nmotion->state, dstate))
        {
            // create a motion
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, dstate);
            motion->parent = nmotion;
            motion->incCost = opt_->motionCost(nmotion->state, motion->state);
            motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);

            // Find nearby neighbors of the new motion
            getNeighbors(motion, nbh);

            rewireTest += nbh.size();
            ++statesGenerated;

            // cache for distance computations
            //
            // Our cost caches only increase in size, so they're only
            // resized if they can't fit the current neighborhood
            if (costs.size() < nbh.size())
            {
                costs.resize(nbh.size());
                incCosts.resize(nbh.size());
                sortedCostIndices.resize(nbh.size());
            }

            // cache for motion validity (only useful in a symmetric space)
            //
            // Our validity caches only increase in size, so they're
            // only resized if they can't fit the current neighborhood
            if (valid.size() < nbh.size())
                valid.resize(nbh.size());
            std::fill(valid.begin(), valid.begin() + nbh.size(), 0);

            // Finding the nearest neighbor to connect to
            // By default, neighborhood states are sorted by cost, and collision checking
            // is performed in increasing order of cost
            if (delayCC_)
            {
                // calculate all costs and distances
                for (std::size_t i = 0; i < nbh.size(); ++i)
                {
                    incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
                    costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
                }
                // sort the nodes
                //
                // we're using index-value pairs so that we can get at
                // original, unsorted indices
                for (std::size_t i = 0; i < nbh.size(); ++i)
                    sortedCostIndices[i] = i;
                std::sort(sortedCostIndices.begin(), sortedCostIndices.begin() + nbh.size(), compareFn);

                // collision check until a valid motion is found
                //
                // ASYMMETRIC CASE: it's possible that none of these
                // neighbors are valid. This is fine, because motion
                // already has a connection to the tree through
                // nmotion (with populated cost fields!).
                for (std::vector<std::size_t>::const_iterator i = sortedCostIndices.begin();
                     i != sortedCostIndices.begin() + nbh.size(); ++i)
                {
                    if (nbh[*i] == nmotion ||
                        ((!useKNearest_ || si_->distance(nbh[*i]->state, motion->state) < maxDistance_) &&
                         si_->checkMotion(nbh[*i]->state, motion->state)))
                    {
                        motion->incCost = incCosts[*i];
                        motion->cost = costs[*i];
                        motion->parent = nbh[*i];
                        valid[*i] = 1;
                        break;
                    }
                    else
                        valid[*i] = -1;
                }
            }
            else  // if not delayCC
            {
                motion->incCost = opt_->motionCost(nmotion->state, motion->state);
                motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);
                // find which one we connect the new state to
                for (std::size_t i = 0; i < nbh.size(); ++i)
                {
                    if (nbh[i] != nmotion)
                    {
                        incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
                        costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
                        if (opt_->isCostBetterThan(costs[i], motion->cost))
                        {
                            if ((!useKNearest_ || si_->distance(nbh[i]->state, motion->state) < maxDistance_) &&
                                si_->checkMotion(nbh[i]->state, motion->state))
                            {
                                motion->incCost = incCosts[i];
                                motion->cost = costs[i];
                                motion->parent = nbh[i];
                                valid[i] = 1;
                            }
                            else
                                valid[i] = -1;
                        }
                    }
                    else
                    {
                        incCosts[i] = motion->incCost;
                        costs[i] = motion->cost;
                        valid[i] = 1;
                    }
                }
            }

            if (useNewStateRejection_)
            {
                if (opt_->isCostBetterThan(solutionHeuristic(motion), bestCost_))
                {
                    nn_->add(motion);
                    motion->parent->children.push_back(motion);
                }
                else  // If the new motion does not improve the best cost it is ignored.
                {
                    si_->freeState(motion->state);
                    delete motion;
                    continue;
                }
            }
            else
            {
                // add motion to the tree
                nn_->add(motion);
                motion->parent->children.push_back(motion);
            }

            if (!heuristic)//meet
            {
                auto *cmotion = new Motion(rev_si_);
                auto* real2 = cmotion->state->as<base::RealVectorStateSpace::StateType>();
                auto* nstate = nmotion->state->as<base::SE2StateSpace::StateType>();
                real2->values[0] = nstate->getX();
                real2->values[1] = nstate->getY();
                Motion* rev_nmotion = rev_nn_->nearest(cmotion);
                double rcheck = rev_si_->distance(cmotion->state, rev_nmotion->state);
                if (rev_nmotion->parent)
                {
                    convertRealVectorToSE2(rev_nmotion, temp, false);
                    if (rcheck < r_meet_ && si_->checkMotion(nmotion->state, temp))
                    {
                        OMPL_INFORM("heuristic");
                        heuristic = true;
                        use_heuristic = true;
                        rev = false;
                        rev_solution = rev_nmotion;
                        while (rev_solution != nullptr)
                        {
                            heuristic_path.push_back(rev_solution);
                            rev_solution = rev_solution->parent;
                        }
                        double duration = time::seconds(time::now() - start_time);
                        OMPL_INFORM("Heuristic time: %f seconds", duration);
                        OMPL_INFORM("num:%ld", heuristic_path.size());
                        for (int i = 0; i < heuristic_path.size() - 1; ++i)
                        {
                            heuristic_idx.push_back(i);
                        }
                    }
                }
            }

            bool checkForSolution = false;
            for (std::size_t i = 0; i < nbh.size(); ++i)
            {
                if (nbh[i] != motion->parent)
                {
                    base::Cost nbhIncCost;
                    if (symCost)
                        nbhIncCost = incCosts[i];
                    else
                        nbhIncCost = opt_->motionCost(motion->state, nbh[i]->state);
                    base::Cost nbhNewCost = opt_->combineCosts(motion->cost, nbhIncCost);
                    if (opt_->isCostBetterThan(nbhNewCost, nbh[i]->cost))
                    {
                        bool motionValid;
                        if (valid[i] == 0)
                        {
                            motionValid =
                                (!useKNearest_ || si_->distance(nbh[i]->state, motion->state) < maxDistance_) &&
                                si_->checkMotion(motion->state, nbh[i]->state);
                        }
                        else
                        {
                            motionValid = (valid[i] == 1);
                        }

                        if (motionValid)
                        {
                            // Remove this node from its parent list
                            removeFromParent(nbh[i]);

                            // Add this node to the new parent
                            nbh[i]->parent = motion;
                            nbh[i]->incCost = nbhIncCost;
                            nbh[i]->cost = nbhNewCost;
                            nbh[i]->parent->children.push_back(nbh[i]);

                            // Update the costs of the node's children
                            updateChildCosts(nbh[i]);

                            checkForSolution = true;
                        }
                    }
                }
            }

            // Add the new motion to the goalMotion_ list, if it satisfies the goal
            double distanceFromGoal;
            if (goal->isSatisfied(motion->state, &distanceFromGoal))
            {
                motion->inGoal = true;
                goalMotions_.push_back(motion);
                checkForSolution = true;
                rev = rev_continue; // 找到解后恢复反向树的生长
                use_heuristic=false;
            }

            // Checking for solution or iterative improvement
            if (checkForSolution)
            {
                bool updatedSolution = false;
                if (!bestGoalMotion_ && !goalMotions_.empty())
                {
                    // We have found our first solution, store it as the best. We only add one
                    // vertex at a time, so there can only be one goal vertex at this moment.
                    bestGoalMotion_ = goalMotions_.front();
                    bestCost_ = bestGoalMotion_->cost;
                    updatedSolution = true;

                    OMPL_INFORM("%s: Found an initial solution with a cost of %.2f in %u iterations (%u "
                                "vertices in the graph)",
                                getName().c_str(), bestCost_.value(), iterations_, nn_->size());
                }
                else
                {
                    // We already have a solution, iterate through the list of goal vertices
                    // and see if there's any improvement.
                    for (auto &goalMotion : goalMotions_)
                    {
                        // Is this goal motion better than the (current) best?
                        if (opt_->isCostBetterThan(goalMotion->cost, bestCost_))
                        {
                            bestGoalMotion_ = goalMotion;
                            bestCost_ = bestGoalMotion_->cost;
                            updatedSolution = true;

                            // Check if it satisfies the optimization objective, if it does, break the for loop
                            if (opt_->isSatisfied(bestCost_))
                            {
                                break;
                            }
                        }
                    }
                }

                if (updatedSolution)
                {
                    if (useTreePruning_)
                    {
                        pruneTree(bestCost_);
                    }

                    if (intermediateSolutionCallback)
                    {
                        std::vector<const base::State *> spath;
                        Motion *intermediate_solution =
                            bestGoalMotion_->parent;  // Do not include goal state to simplify code.

                        // Push back until we find the start, but not the start itself
                        while (intermediate_solution->parent != nullptr)
                        {
                            spath.push_back(intermediate_solution->state);
                            intermediate_solution = intermediate_solution->parent;
                        }

                        intermediateSolutionCallback(this, spath, bestCost_);
                    }
                }
            }

            // Checking for approximate solution (closest state found to the goal)
            if (goalMotions_.size() == 0 && distanceFromGoal < approxDist)
            {
                approxGoalMotion = motion;
                approxDist = distanceFromGoal;
            }
        }

        // terminate if a sufficient solution is found
        if (bestGoalMotion_ && opt_->isSatisfied(bestCost_))
            break;
    }

    // Add our solution (if it exists)
    Motion *newSolution = nullptr;
    if (bestGoalMotion_)
    {
        // We have an exact solution
        newSolution = bestGoalMotion_;
    }
    else if (approxGoalMotion)
    {
        // We don't have a solution, but we do have an approximate solution
        newSolution = approxGoalMotion;
    }
    // No else, we have nothing

    // Add what we found
    if (newSolution)
    {
        ptc.terminate();
        // construct the solution path
        std::vector<Motion *> mpath;
        Motion *iterMotion = newSolution;
        while (iterMotion != nullptr)
        {
            mpath.push_back(iterMotion);
            iterMotion = iterMotion->parent;
        }

        // set the solution path
        auto path(std::make_shared<PathGeometric>(si_));
        for (int i = mpath.size() - 1; i >= 0; --i)
            path->append(mpath[i]->state);

        // Add the solution path.
        base::PlannerSolution psol(path);
        psol.setPlannerName(getName());

        // If we don't have a goal motion, the solution is approximate
        if (!bestGoalMotion_)
            psol.setApproximate(approxDist);

        // Does the solution satisfy the optimization objective?
        psol.setOptimized(opt_, newSolution->cost, opt_->isSatisfied(bestCost_));
        pdef_->addSolutionPath(psol);
    }
    // No else, we have nothing

    si_->freeState(xstate);
    si_->freeState(temp);
    if (rmotion->state != nullptr)
        si_->freeState(rmotion->state);
    delete rmotion;

    rev_si_->freeState(rev_xstate);
    if (rev_rmotion->state != nullptr)
        rev_si_->freeState(rev_rmotion->state);
    delete rev_rmotion;

    OMPL_INFORM("%s: Created %u new states. Checked %u rewire options. %u goal states in tree. Final solution cost "
                "%.3f",
                getName().c_str(), statesGenerated, rewireTest, goalMotions_.size(), bestCost_.value());

    // We've added a solution if newSolution == true, and it is an approximate solution if bestGoalMotion_ == false
    return {newSolution != nullptr, bestGoalMotion_ == nullptr};
}

void ompl::geometric::BiRRTstar::getNeighbors(Motion *motion, std::vector<Motion *> &nbh) const
{
    auto cardDbl = static_cast<double>(nn_->size() + 1u);
    if (useKNearest_)
    {
        //- k-nearest RRT*
        unsigned int k = std::ceil(k_rrt_ * log(cardDbl));
        nn_->nearestK(motion, k, nbh);
    }
    else
    {
        double r = std::min(
            maxDistance_, r_rrt_ * std::pow(log(cardDbl) / cardDbl, 1 / static_cast<double>(si_->getStateDimension())));
        nn_->nearestR(motion, r, nbh);
    }
}

void ompl::geometric::BiRRTstar::rev_getNeighbors(Motion *motion, std::vector<Motion *> &nbh) const
{
    auto cardDbl = static_cast<double>(rev_nn_->size() + 1u);
    if (useKNearest_)
    {
        //- k-nearest RRT*
        unsigned int k = std::ceil(k_rrt_ * log(cardDbl));
        rev_nn_->nearestK(motion, k, nbh);
    }
    else
    {
        double r = std::min(
            maxDistance_, r_rrt_ * std::pow(log(cardDbl) / cardDbl, 1 / static_cast<double>(si_->getStateDimension())));
        rev_nn_->nearestR(motion, r, nbh);
    }
}

void ompl::geometric::BiRRTstar::removeFromParent(Motion *m)
{
    for (auto it = m->parent->children.begin(); it != m->parent->children.end(); ++it)
    {
        if (*it == m)
        {
            m->parent->children.erase(it);
            break;
        }
    }
}

void ompl::geometric::BiRRTstar::updateChildCosts(Motion *m)
{
    for (std::size_t i = 0; i < m->children.size(); ++i)
    {
        m->children[i]->cost = opt_->combineCosts(m->cost, m->children[i]->incCost);
        updateChildCosts(m->children[i]);
    }
}

void ompl::geometric::BiRRTstar::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state)
                si_->freeState(motion->state);
            delete motion;
        }
    }
}

void ompl::geometric::BiRRTstar::rev_freeMemory()
{
    if (rev_nn_)
    {
        std::vector<Motion *> rev_motions;
        rev_nn_->list(rev_motions);
        for (auto &motion : rev_motions)
        {
            if (motion->state != nullptr)
                rev_si_->freeState(motion->state);
            delete motion;
        }
    }
    heuristic_path.clear();
    heuristic_idx.clear();
}

void ompl::geometric::BiRRTstar::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    if (nn_)
        nn_->list(motions);

    if (bestGoalMotion_)
        data.addGoalVertex(base::PlannerDataVertex(bestGoalMotion_->state));

    for (auto &motion : motions)
    {
        if (motion->parent == nullptr)
            data.addStartVertex(base::PlannerDataVertex(motion->state));
        else
            data.addEdge(base::PlannerDataVertex(motion->parent->state), base::PlannerDataVertex(motion->state));
    }
}

int ompl::geometric::BiRRTstar::pruneTree(const base::Cost &pruneTreeCost)
{
    // Variable
    // The percent improvement (expressed as a [0,1] fraction) in cost
    double fracBetter;
    // The number pruned
    int numPruned = 0;

    if (opt_->isFinite(prunedCost_))
    {
        fracBetter = std::abs((pruneTreeCost.value() - prunedCost_.value()) / prunedCost_.value());
    }
    else
    {
        fracBetter = 1.0;
    }

    if (fracBetter > pruneThreshold_)
    {
        // We are only pruning motions if they, AND all descendents, have a estimated cost greater than pruneTreeCost
        // The easiest way to do this is to find leaves that should be pruned and ascend up their ancestry until a
        // motion is found that is kept.
        // To avoid making an intermediate copy of the NN structure, we process the tree by descending down from the
        // start(s).
        // In the first pass, all Motions with a cost below pruneTreeCost, or Motion's with children with costs below
        // pruneTreeCost are added to the replacement NN structure,
        // while all other Motions are stored as either a 'leaf' or 'chain' Motion. After all the leaves are
        // disconnected and deleted, we check
        // if any of the the chain Motions are now leaves, and repeat that process until done.
        // This avoids (1) copying the NN structure into an intermediate variable and (2) the use of the expensive
        // NN::remove() method.

        // Variable
        // The queue of Motions to process:
        std::queue<Motion *, std::deque<Motion *>> motionQueue;
        // The list of leaves to prune
        std::queue<Motion *, std::deque<Motion *>> leavesToPrune;
        // The list of chain vertices to recheck after pruning
        std::list<Motion *> chainsToRecheck;

        // Clear the NN structure:
        nn_->clear();

        // Put all the starts into the NN structure and their children into the queue:
        // We do this so that start states are never pruned.
        for (auto &startMotion : startMotions_)
        {
            // Add to the NN
            nn_->add(startMotion);

            // Add their children to the queue:
            addChildrenToList(&motionQueue, startMotion);
        }

        while (motionQueue.empty() == false)
        {
            // Test, can the current motion ever provide a better solution?
            if (keepCondition(motionQueue.front(), pruneTreeCost))
            {
                // Yes it can, so it definitely won't be pruned
                // Add it back into the NN structure
                nn_->add(motionQueue.front());

                // Add it's children to the queue
                addChildrenToList(&motionQueue, motionQueue.front());
            }
            else
            {
                // No it can't, but does it have children?
                if (motionQueue.front()->children.empty() == false)
                {
                    // Yes it does.
                    // We can minimize the number of intermediate chain motions if we check their children
                    // If any of them won't be pruned, then this motion won't either. This intuitively seems
                    // like a nice balance between following the descendents forever.

                    // Variable
                    // Whether the children are definitely to be kept.
                    bool keepAChild = false;

                    // Find if any child is definitely not being pruned.
                    for (unsigned int i = 0u; keepAChild == false && i < motionQueue.front()->children.size(); ++i)
                    {
                        // Test if the child can ever provide a better solution
                        keepAChild = keepCondition(motionQueue.front()->children.at(i), pruneTreeCost);
                    }

                    // Are we *definitely* keeping any of the children?
                    if (keepAChild)
                    {
                        // Yes, we are, so we are not pruning this motion
                        // Add it back into the NN structure.
                        nn_->add(motionQueue.front());
                    }
                    else
                    {
                        // No, we aren't. This doesn't mean we won't though
                        // Move this Motion to the temporary list
                        chainsToRecheck.push_back(motionQueue.front());
                    }

                    // Either way. add it's children to the queue
                    addChildrenToList(&motionQueue, motionQueue.front());
                }
                else
                {
                    // No, so we will be pruning this motion:
                    leavesToPrune.push(motionQueue.front());
                }
            }

            // Pop the iterator, std::list::erase returns the next iterator
            motionQueue.pop();
        }

        // We now have a list of Motions to definitely remove, and a list of Motions to recheck
        // Iteratively check the two lists until there is nothing to to remove
        while (leavesToPrune.empty() == false)
        {
            // First empty the current leaves-to-prune
            while (leavesToPrune.empty() == false)
            {
                // If this leaf is a goal, remove it from the goal set
                if (leavesToPrune.front()->inGoal == true)
                {
                    // Warn if pruning the _best_ goal
                    if (leavesToPrune.front() == bestGoalMotion_)
                    {
                        OMPL_ERROR("%s: Pruning the best goal.", getName().c_str());
                    }
                    // Remove it
                    goalMotions_.erase(std::remove(goalMotions_.begin(), goalMotions_.end(), leavesToPrune.front()),
                                       goalMotions_.end());
                }

                // Remove the leaf from its parent
                removeFromParent(leavesToPrune.front());

                // Erase the actual motion
                // First free the state
                si_->freeState(leavesToPrune.front()->state);

                // then delete the pointer
                delete leavesToPrune.front();

                // And finally remove it from the list, erase returns the next iterator
                leavesToPrune.pop();

                // Update our counter
                ++numPruned;
            }

            // Now, we need to go through the list of chain vertices and see if any are now leaves
            auto mIter = chainsToRecheck.begin();
            while (mIter != chainsToRecheck.end())
            {
                // Is the Motion a leaf?
                if ((*mIter)->children.empty() == true)
                {
                    // It is, add to the removal queue
                    leavesToPrune.push(*mIter);

                    // Remove from this queue, getting the next
                    mIter = chainsToRecheck.erase(mIter);
                }
                else
                {
                    // Is isn't, skip to the next
                    ++mIter;
                }
            }
        }

        // Now finally add back any vertices left in chainsToReheck.
        // These are chain vertices that have descendents that we want to keep
        for (const auto &r : chainsToRecheck)
            // Add the motion back to the NN struct:
            nn_->add(r);

        // All done pruning.
        // Update the cost at which we've pruned:
        prunedCost_ = pruneTreeCost;

        // And if we're using the pruned measure, the measure to which we've pruned
        if (usePrunedMeasure_)
        {
            prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);

            if (useKNearest_ == false)
            {
                calculateRewiringLowerBounds();
            }
        }
        // No else, prunedMeasure_ is the si_ measure by default.
    }

    return numPruned;
}

void ompl::geometric::BiRRTstar::addChildrenToList(std::queue<Motion *, std::deque<Motion *>> *motionList, Motion *motion)
{
    for (auto &child : motion->children)
    {
        motionList->push(child);
    }
}

bool ompl::geometric::BiRRTstar::keepCondition(const Motion *motion, const base::Cost &threshold) const
{
    // We keep if the cost-to-come-heuristic of motion is <= threshold, by checking
    // if !(threshold < heuristic), as if b is not better than a, then a is better than, or equal to, b
    if (bestGoalMotion_ && motion == bestGoalMotion_)
    {
        // If the threshold is the theoretical minimum, the bestGoalMotion_ will sometimes fail the test due to floating point precision. Avoid that.
        return true;
    }

    return !opt_->isCostBetterThan(threshold, solutionHeuristic(motion));
}

ompl::base::Cost ompl::geometric::BiRRTstar::solutionHeuristic(const Motion *motion) const
{
    base::Cost costToCome;
    if (useAdmissibleCostToCome_)
    {
        // Start with infinite cost
        costToCome = opt_->infiniteCost();

        // Find the min from each start
        for (auto &startMotion : startMotions_)
        {
            costToCome = opt_->betterCost(
                costToCome, opt_->motionCost(startMotion->state,
                                             motion->state));  // lower-bounding cost from the start to the state
        }
    }
    else
    {
        costToCome = motion->cost;  // current cost from the state to the goal
    }

    const base::Cost costToGo =
        opt_->costToGo(motion->state, pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
    return opt_->combineCosts(costToCome, costToGo);            // add the two costs
}

void ompl::geometric::BiRRTstar::setTreePruning(const bool prune)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // If we just disabled tree pruning, but we wee using prunedMeasure, we need to disable that as it required myself
    if (prune == false && getPrunedMeasure() == true)
    {
        setPrunedMeasure(false);
    }

    // Store
    useTreePruning_ = prune;
}

void ompl::geometric::BiRRTstar::setPrunedMeasure(bool informedMeasure)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option only works with informed sampling
    if (informedMeasure == true && (useInformedSampling_ == false || useTreePruning_ == false))
    {
        OMPL_ERROR("%s: InformedMeasure requires InformedSampling and TreePruning.", getName().c_str());
    }

    // Check if we're changed and update parameters if we have:
    if (informedMeasure != usePrunedMeasure_)
    {
        // Store the setting
        usePrunedMeasure_ = informedMeasure;

        // Update the prunedMeasure_ appropriately, if it has been configured.
        if (setup_ == true)
        {
            if (usePrunedMeasure_)
            {
                prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);
            }
            else
            {
                prunedMeasure_ = si_->getSpaceMeasure();
            }
        }

        // And either way, update the rewiring radius if necessary
        if (useKNearest_ == false)
        {
            calculateRewiringLowerBounds();
        }
    }
}

void ompl::geometric::BiRRTstar::setInformedSampling(bool informedSampling)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option is mutually exclusive with setSampleRejection, assert that:
    if (informedSampling == true && useRejectionSampling_ == true)
    {
        OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
    }

    // If we just disabled tree pruning, but we are using prunedMeasure, we need to disable that as it required myself
    if (informedSampling == false && getPrunedMeasure() == true)
    {
        setPrunedMeasure(false);
    }

    // Check if we're changing the setting of informed sampling. If we are, we will need to create a new sampler, which
    // we only want to do if one is already allocated.
    if (informedSampling != useInformedSampling_)
    {
        // If we're disabled informedSampling, and prunedMeasure is enabled, we need to disable that
        if (informedSampling == false && usePrunedMeasure_ == true)
        {
            setPrunedMeasure(false);
        }

        // Store the value
        useInformedSampling_ = informedSampling;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void ompl::geometric::BiRRTstar::setSampleRejection(const bool reject)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option is mutually exclusive with setInformedSampling, assert that:
    if (reject == true && useInformedSampling_ == true)
    {
        OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
    }

    // Check if we're changing the setting of rejection sampling. If we are, we will need to create a new sampler, which
    // we only want to do if one is already allocated.
    if (reject != useRejectionSampling_)
    {
        // Store the setting
        useRejectionSampling_ = reject;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void ompl::geometric::BiRRTstar::setOrderedSampling(bool orderSamples)
{
    // Make sure we're using some type of informed sampling
    if (useInformedSampling_ == false && useRejectionSampling_ == false)
    {
        OMPL_ERROR("%s: OrderedSampling requires either informed sampling or rejection sampling.", getName().c_str());
    }

    // Check if we're changing the setting. If we are, we will need to create a new sampler, which we only want to do if
    // one is already allocated.
    if (orderSamples != useOrderedSampling_)
    {
        // Store the setting
        useOrderedSampling_ = orderSamples;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void ompl::geometric::BiRRTstar::allocSampler()
{
    // Allocate the appropriate type of sampler.
    if (useInformedSampling_)
    {
        // We are using informed sampling, this can end-up reverting to rejection sampling in some cases
        OMPL_INFORM("%s: Using informed sampling.", getName().c_str());
        infSampler_ = opt_->allocInformedStateSampler(pdef_, numSampleAttempts_);
    }
    else if (useRejectionSampling_)
    {
        // We are explicitly using rejection sampling.
        OMPL_INFORM("%s: Using rejection sampling.", getName().c_str());
        infSampler_ = std::make_shared<base::RejectionInfSampler>(pdef_, numSampleAttempts_);
    }
    else
    {
        // We are using a regular sampler
        sampler_ = si_->allocStateSampler();
    }

    // Wrap into a sorted sampler
    if (useOrderedSampling_ == true)
    {
        infSampler_ = std::make_shared<base::OrderedInfSampler>(infSampler_, batchSize_);
    }
    // No else
}

bool ompl::geometric::BiRRTstar::sampleUniform(base::State *statePtr)
{
    // Use the appropriate sampler
    if (useInformedSampling_ || useRejectionSampling_)
    {
        // Attempt the focused sampler and return the result.
        // If bestCost is changing a lot by small amounts, this could
        // be prunedCost_ to reduce the number of times the informed sampling
        // transforms are recalculated.
        return infSampler_->sampleUniform(statePtr, bestCost_);
    }
    else
    {
        // Simply return a state from the regular sampler
        sampler_->sampleUniform(statePtr);

        // Always true
        return true;
    }
}

void ompl::geometric::BiRRTstar::calculateRewiringLowerBounds()
{
    const auto dimDbl = static_cast<double>(si_->getStateDimension());

    // k_rrt > 2^(d + 1) * e * (1 + 1 / d).  K-nearest RRT*
    k_rrt_ = rewireFactor_ * (std::pow(2, dimDbl + 1) * boost::math::constants::e<double>() * (1.0 + 1.0 / dimDbl));

    // r_rrt > (2*(1+1/d))^(1/d)*(measure/ballvolume)^(1/d)
    // If we're not using the informed measure, prunedMeasure_ will be set to si_->getSpaceMeasure();
    r_rrt_ =
        rewireFactor_ *
        std::pow(2 * (1.0 + 1.0 / dimDbl) * (prunedMeasure_ / unitNBallMeasure(si_->getStateDimension())), 1.0 / dimDbl);
}

bool ompl::geometric::BiRRTstar::revcheckMotion(const base::State *s1, const base::State *s2)
{
    if (!rev_si_->isValid(s2))
    {
        return false;
    }
    bool result = true;
    // int nd = rev_stateSpace_->validSegmentCount(s1, s2);
    // ROS_INFO("nd:%d",nd);
    int nd =100;
    /* initialize the queue of test positions */
    std::queue<std::pair<int, int>> pos;
    if (nd >= 2)
    {
        pos.emplace(1, nd - 1);

        /* temporary storage for the checked state */
        base::State *test = rev_si_->allocState();

        /* repeatedly subdivide the path segment in the middle (and check the middle) */
        while (!pos.empty())
        {
            std::pair<int, int> x = pos.front();

            int mid = (x.first + x.second) / 2;
            rev_stateSpace_->interpolate(s1, s2, (double)mid / (double)nd, test);

            if (!rev_si_->isValid(test))
            {
                result = false;
                break;
            }

            pos.pop();

            if (x.first < mid)
                pos.emplace(x.first, mid - 1);
            if (x.second > mid)
                pos.emplace(mid + 1, x.second);
        }

        rev_si_->freeState(test);
    }
    return result;
}

void ompl::geometric::BiRRTstar::convertRealVectorToSE2(const Motion* rev_motion, base::State* se2_state, bool disturb) {
    // 获取rev_motion和其父节点的位置
    const auto *rev_state = rev_motion->state->as<base::RealVectorStateSpace::StateType>();
    const auto *rev_parent_state = rev_motion->parent->state->as<base::RealVectorStateSpace::StateType>();

    // 计算朝向父节点的角度
    double dx = rev_parent_state->values[0] - rev_state->values[0];
    double dy = rev_parent_state->values[1] - rev_state->values[1];
    double yaw = atan2(dy, dx);

    std::random_device rd; 
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, M_PI/3);

    auto* se2 = se2_state->as<base::SE2StateSpace::StateType>();

    if(disturb)
    {
        double c = sqrt(dx * dx + dy * dy) / 2;
        double a = c + maxDistance_ * rev_stepsize / n_;
        double b = sqrt(a * a - c * c);
        std::pair<double, double> point = random_point_in_ellipse(a, b);
        se2->setX((rev_state->values[0] + rev_parent_state->values[0]) / 2 + point.first);
        se2->setY((rev_state->values[1] + rev_parent_state->values[1]) / 2 + point.second);
        yaw = rng_.uniformReal(-M_PI, M_PI);
        se2->setYaw(yaw);
        // std::pair<double,double> point = randomPointInCircle(maxDistance_/2);
        // se2->setX(rev_state->values[0]+point.first);
        // se2->setY(rev_state->values[1]+point.second);
    }
    else{
        // 转换为SE2状态      
        se2->setX(rev_state->values[0]);
        se2->setY(rev_state->values[1]);
        se2->setYaw(yaw);
    }  
}

std::vector<std::pair<double,double>> ompl::geometric::BiRRTstar::getheuristicData()
{
    std::vector<std::pair<double,double>> points;
    if (!heuristic_path.empty()) {
        for (auto motion : heuristic_path) {
            const auto* state = motion->state->as<base::RealVectorStateSpace::StateType>();
            points.emplace_back(state->values[0], state->values[1]);
            // OMPL_INFORM("%f,%f",state->values[0], state->values[1]);
        }
    }
    return points;
}