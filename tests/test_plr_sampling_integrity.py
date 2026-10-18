"""PLR priority freshness, eligible replay, delayed feedback and controls."""
import numpy as np
import pytest
from AAMAS_Comp.curriculum.plr import PLRBuffer, _LevelEntry
from AAMAS_Comp.envs.ns_env_sampler import NS_ENV_SAMPLERS


def buffer(**kwargs):
    return PLRBuffer(NS_ENV_SAMPLERS['frozenlake'](seed=8),seed=9,**kwargs)


def entries(b, scores):
    b._entries=[_LevelEntry(config=b.sampler.sample(),score=score,n_episodes=1) for score in scores]


def test_latest_score_replaces_obsolete_history():
    b=buffer(staleness_coef=0)
    entries(b,[0,0])
    for _ in range(100): b.update(0,1)
    for _ in range(100): b.update(0,0)
    b.update(1,.2)
    assert b._entries[0].score==0
    assert b.sampling_distribution()[1]>.99


def test_explicit_legacy_average_remains_available():
    b=buffer(score_ema_alpha=None)
    b._entries=[_LevelEntry(config=b.sampler.sample())]
    for _ in range(100): b.update(0,1)
    for _ in range(100): b.update(0,0)
    assert b._entries[0].score==pytest.approx(.5)


def test_first_ema_score_is_not_diluted_with_initial_zero():
    b=buffer(score_ema_alpha=.1)
    b._entries=[_LevelEntry(config=b.sampler.sample())]
    b.update(0,10);assert b._entries[0].score==10
    b.update(0,0);assert b._entries[0].score==9


@pytest.mark.parametrize('uniform',[False,True])
def test_unscored_levels_cannot_be_replayed_or_mutation_parents(uniform):
    b=buffer(capacity=20,min_fill=.1,replay_prob=1,uniform_replay=uniform)
    entries(b,[1,.1,999])
    b._entries[2].n_episodes=0
    assert b.sampling_distribution()[2]==0
    for _ in range(200):assert b.sample()[0]!=2
    # Mutation samples from the same eligibility distribution.
    b.replay_prob=0;b.mutation_prob=1;b.capacity=1000
    parents=[]
    b.sampler.mutate=lambda config,*args:parents.append(config) or config
    excluded=b._entries[2].config
    for _ in range(20):b.sample()
    assert parents and all(p is not excluded for p in parents)


def test_warmup_requires_scored_levels_and_no_feedback_can_deadlock_collector():
    b=buffer(capacity=5,min_fill=.4,replay_prob=1,mutation_prob=1)
    for _ in range(5):
        b.sample()
        assert not b.last_was_replay and not b.last_was_mutation
    assert not b.is_warm
    np.testing.assert_array_equal(b.sampling_distribution(),np.zeros(5))
    # Buffer full while learner is waiting for collection: uniform forced
    # re-evaluation is explicitly marked and protects every pending level.
    slots=[]
    for _ in range(500):
        slots.append(b.sample()[0])
        assert b.last_was_forced_replay and b.last_was_replay
        assert not b.last_was_mutation
    assert len(set(slots))==5
    assert max(np.bincount(slots))/len(slots)<.3
    assert b.stats()['plr/unscored_evictions']==0
    b.update(0,1);assert not b.is_warm
    # Before the scored warm-up threshold, preserve the scores that did
    # arrive instead of repeatedly evicting them and restarting warm-up.
    for _ in range(30):
        assert b.sample()[0]!=0
        assert b.last_was_forced_replay
        assert b._entries[0].score==1 and b._entries[0].n_episodes==1
    b.update(1,.1);assert b.is_warm
    for _ in range(50): assert b.sample()[0] in (0,1)


def test_staleness_counts_selections_not_delayed_feedback():
    b=buffer(capacity=2,min_fill=.5,replay_prob=1,uniform_replay=True)
    entries(b,[1,2])
    for _ in range(10):
        before=[e.staleness for e in b._entries]
        selected,_=b.sample()
        assert b._entries[selected].staleness==0
        other=1-selected
        assert b._entries[other].staleness==before[other]+1
    ages=[e.staleness for e in b._entries]
    for _ in range(5): b.update(other,3)
    assert [e.staleness for e in b._entries]==ages


@pytest.mark.parametrize('temperature',[.1,1e-8])
def test_equal_scores_have_equal_priority_without_position_bias(temperature):
    b=buffer(staleness_coef=0,score_temp=temperature)
    entries(b,[0,0,0])
    np.testing.assert_allclose(b.sampling_distribution(),[1/3]*3)
    b._entries[1].n_episodes=0
    np.testing.assert_allclose(b.sampling_distribution(),[.5,0,.5])


def test_uniform_control_eviction_does_not_depend_on_scores():
    a=buffer(capacity=3,uniform_replay=True);c=buffer(capacity=3,uniform_replay=True)
    entries(a,[0,1,2]);entries(c,[100,10,0])
    chosen=[]
    for _ in range(100):
        # Restore score observations; compare identical RNG draws under
        # different score orderings and ensure every slot can be evicted.
        for i,e in enumerate(a._entries):e.n_episodes=1;e.score=i
        for i,e in enumerate(c._entries):e.n_episodes=1;e.score=100-10*i
        ia,_=a._explore();ic,_=c._explore()
        assert ia==ic;chosen.append(ia)
    assert set(chosen)=={0,1,2}


@pytest.mark.parametrize('score',[-1,float('nan'),float('inf')])
def test_invalid_scores_cannot_poison_distribution(score):
    b=buffer();entries(b,[1])
    with pytest.raises(ValueError):b.update(0,score)
    assert b._entries[0].score==1
