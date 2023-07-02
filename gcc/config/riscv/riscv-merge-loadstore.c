/* Merge load/store pass for RISC-V.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

GCC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING3.  If not see
<http://www.gnu.org/licenses/>.  */

#define IN_TARGET_CODE 1

#define INCLUDE_ALGORITHM
#define INCLUDE_FUNCTIONAL
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "cfgrtl.h"
#include "backend.h"
#include "rtl.h"
#include "df.h"
#include "rtl-ssa.h"
#include "tm.h"
#include "regs.h"
#include "target.h"
#include "memmodel.h"
#include "emit-rtl.h"
#include "print-rtl.h"
#include "pretty-print.h"
#include "predict.h"
#include "tree-pass.h"
#include "cfgcleanup.h"
#include "recog.h"

namespace {

const pass_data pass_data_merge_loadstores =
{
  RTL_PASS, /* type */
  "merge_loadstores", /* name */
  OPTGROUP_NONE, /* optinfo_flags */
  TV_NONE, /* tv_id */
  0, /* properties_required */
  0, /* properties_provided */
  0, /* properties_destroyed */
  0, /* todo_flags_start */
  0, /* todo_flags_finish */
};

/* Represent groups of scalar insns that can be replaced with vector insn. */
struct grouped_insns {
  vec<vec<rtx_insn *>> insns;
  auto_vec<machine_mode> scalar_modes;
  auto_vec<machine_mode> vector_modes;
  bool load;
  grouped_insns (bool load_not_store) {
    load = load_not_store;
    insns = vNULL;
  }
  ~grouped_insns () {
    for (int i = 0; i< insns.length (); i++)
      insns[i].release ();
    insns.release ();
  }
};

/* New pass - merge loadstores */
class pass_merge_loadstores : public rtl_opt_pass
{
public:
  pass_merge_loadstores (gcc::context *ctxt)
    : rtl_opt_pass (pass_data_merge_loadstores, ctxt)
  {}

  /* opt_pass methods: */
  virtual bool gate (function *)
    {
      return TARGET_RVC && optimize > 0;
    }
  virtual unsigned int execute (function *);

private:
  typedef auto_vec<rtx_insn *> insn_vec;
  typedef hash_map<rtx_insn *, rtl_ssa::insn_info *> insn_to_insn_info_map;

  void slp_bb (basic_block bb);
  void assign_group_id_to_insn (basic_block bb, insn_vec &insn_candidates,
      vec<int> &insn_groups);
  bool is_candidate (rtx_insn *insnn, insn_vec &insn_candidates, 
      vec<int> &insn_groups, int group_id);
  bool slp_analyze_bb (insn_vec &insn_candidates, vec<int> &insn_groups,
      struct grouped_insns *grouped_loads, struct grouped_insns *grouped_stores);
  void pair_insns (insn_vec &insn_candidates, vec<int> &insn_groups,
      struct grouped_insns *grouped_loads, struct grouped_insns *grouped_stores);
  bool adjacent_addresses (struct address_info *addr_info_a, 
      struct address_info *addr_info_b, machine_mode *s_mode, machine_mode *v_mode);
  bool adjacent (rtx_insn *i_a, rtx_insn *i_b, bool *load, machine_mode *mode, 
      machine_mode *v_mode);
  bool already_paired (vec<std::pair<int,int>> *paired_insns, int left_id, int right_id);
  bool merge_full_groups (struct grouped_insns *grouped_insns);
  int grouped_ins (rtx_insn *in, struct grouped_insns *grouped_insns);
  bool try_to_schedule_ins (rtl_ssa::insn_info *in, insn_to_insn_info_map &scheduled);
  bool group_all_scheduled (insn_to_insn_info_map &scheduled, 
      struct grouped_insns *grouped_insns, int grouped_index);
  int matching_store_group (insn_to_insn_info_map &scheduled, vec<rtx_insn *> &loads,
      struct grouped_insns *grouped_stores, int full_group_size);
  void transform (struct grouped_insns *grouped_insns, int group_idx, 
      rtl_ssa::insn_info *insn_to_change, basic_block bb);
  bool swap_with_SIMD (rtl_ssa::insn_info *insn_to_change, bool load, 
      machine_mode new_mode);
}; // class pass_merge_loadstores

/* Change scalar insn to vectorized insn. */
bool pass_merge_loadstores::swap_with_SIMD (rtl_ssa::insn_info *insn_to_change, 
  bool load, machine_mode new_mode) {
  rtx_insn *insn_to_change_rtx = insn_to_change->rtl ();
  auto attempt = crtl->ssa->new_change_attempt ();
  rtl_ssa::insn_change change (insn_to_change);
  if (!rtl_ssa::restrict_movement (change))
    return false;
  insn_change_watermark watermark;
  /* We have already checked that insn candidates are SET insns with the following forms:
   (set (mem ... )
        (subreg ... )
   ) - for stores,
   (set (reg ... )
        (zero_extend ... (mem ... ))
   ) - for loads. */
  rtx old_pat = PATTERN (insn_to_change_rtx);
  rtx reg = load? SET_DEST (old_pat):
            /* For stores, src register is the first element in SUBREG rtx. */ 
            XEXP (SET_SRC (old_pat), 0);
  rtx old_mem = load? XEXP (SET_SRC (old_pat),0): SET_DEST (old_pat);
  rtx addr = XEXP (old_mem,0);
  rtx new_mem = gen_rtx_MEM (new_mode, addr);
  if  (MEM_READONLY_P (old_mem))
    new_mem = gen_const_mem (new_mode, addr);
  rtx new_pat = load? gen_rtx_SET (reg,new_mem) : gen_rtx_SET (new_mem,reg);
  validate_change (insn_to_change_rtx, &PATTERN (insn_to_change_rtx), new_pat, 1);
  if (!recog (attempt, change) || !rtl_ssa::change_is_worthwhile (change)){
    cancel_changes (0);
    return false;
  }
  confirm_change_group ();
  rtl_ssa::insn_change *changes [] = {&change};
  if (!crtl->ssa->verify_insn_changes (changes))
    return false;
  crtl->ssa->change_insn (change);
  return true;
}

/* Swap group with SIMD insn. Change the first insn in the group and delete 
   others. */
void pass_merge_loadstores::transform (struct grouped_insns *grouped_insns, int group_idx, 
  rtl_ssa::insn_info *insn_to_change, basic_block bb) {
  machine_mode new_mode = grouped_insns->vector_modes[group_idx];
  bool load_nt_store = grouped_insns->load;
  if (!swap_with_SIMD (insn_to_change, load_nt_store, new_mode))
     return;
  
  /* Delete other grouped insns. */
  rtx_insn *insn;
  FOR_BB_INSNS (bb,insn) {
    for (size_t i = 1; i< grouped_insns->insns[group_idx].length (); i++)
      if (INSN_UID (grouped_insns->insns[group_idx][i]) == INSN_UID (insn)) {
        delete_insn (insn);
        update_bb_for_insn (bb);
      }
  } 
}

/* For grouped loads, we need to ensure that there is an appropriate group of
   store insns. */
int pass_merge_loadstores::matching_store_group (insn_to_insn_info_map &scheduled,
vec<rtx_insn *> &loads, struct grouped_insns *grouped_stores, int full_group_size) {
  for (size_t i = 0; i < grouped_stores->insns.length (); i++) {
    if (grouped_stores->insns[i].length () != loads.length ())
      continue;
    int cnt = 0;
    for (size_t j = 0; j < loads.length (); j++) {
      rtl_ssa::insn_info *load_info = *(scheduled.get (loads[j]));
      if (load_info->num_defs () != 1 || 
          !is_a<rtl_ssa::set_info *> (load_info->defs ()[0]))
        return -1;
      rtl_ssa::set_info *load_set_info =  dyn_cast <rtl_ssa::set_info *> 
                                          (load_info->defs ()[0]);
      rtl_ssa::use_info *use_of_load = load_set_info->single_nondebug_insn_use ();
      if (!use_of_load)
        return -1;
      if (grouped_stores->insns[i][j] == use_of_load->insn ()->rtl ()) {
        ++cnt;
        if (cnt == full_group_size)
          return i;
      }
    }
  }
  return -1;
}

/* Check if every insn in the group is scheduled. */
bool pass_merge_loadstores::group_all_scheduled (insn_to_insn_info_map &scheduled,
  struct grouped_insns *grouped_insns, int grouped_index) {
  for (size_t i = 0; i < grouped_insns->insns[grouped_index].length (); i++)
    if (!scheduled.get (grouped_insns->insns[grouped_index][i]))
        return false;
  return true;
}

/* Each insn is scheduled as soon as all statements on which it is dependent 
  have been scheduled. */
bool pass_merge_loadstores::try_to_schedule_ins (rtl_ssa::insn_info *in, 
  insn_to_insn_info_map &scheduled) {
  for (rtl_ssa::use_info *use : in->uses ()){
    if (!use->def () || use->def ()->is_artificial ())
      continue;
    /* Insn can't be scheduled now */
    if (!scheduled.get (use->def ()->insn ()->rtl ()))
      return false;
  }
  scheduled.put (in->rtl (), in);
  return true;
}

/* Check if insn is grouped insn. Return group index, or -1 if it's not. */
int pass_merge_loadstores::grouped_ins (rtx_insn *in, struct grouped_insns *grouped_insns) {
  for (int i = 0; i< grouped_insns->insns.length () ; i++)
    for (size_t j = 0; j<grouped_insns->insns[i].length (); j++)
      if (grouped_insns->insns[i][j] == in)
        return i;
  return -1;
}

/* FORNOW: We only group insns that operate on the same data widths. */
bool pass_merge_loadstores::merge_full_groups (struct grouped_insns *grouped_insns) {
  if (grouped_insns->insns.length () == 0)
    return false;
  bool full_group = false;

  /* We're comparing every two groups and trying to merge them, based
     on the last insn in one and the first insn in another group. */
  for (size_t i = 0; i< grouped_insns->insns.length () - 1; i++) {
    int group_length = grouped_insns->insns[i].length ();

    /* During merging there could appear empty groups. */
    if (group_length == 0)
      continue;
    machine_mode scalar_mode = grouped_insns->scalar_modes[i];
    machine_mode vector_mode = grouped_insns->vector_modes[i];

    /* FULL_GROUP_SIZE represents the maximum number of insns in the 
       group we can merge. */
    int full_group_size = GET_MODE_SIZE (vector_mode) 
                        / GET_MODE_SIZE (scalar_mode);

    /* Last insn in group I. */
    int id_right = INSN_UID (grouped_insns->insns[i][group_length - 1]);
    for (size_t j = i + 1; j<grouped_insns->insns.length (); j++) {
      if (grouped_insns->insns[j].length () <= 1)
        continue;       

      /* First insn in group J. */
      int id_left = INSN_UID (grouped_insns->insns[j][0]);

      /* Merging groups. */
      if (id_right == id_left && group_length < full_group_size) {
        grouped_insns->insns[i].safe_push (grouped_insns->insns[j][1]);
        grouped_insns->insns[j].truncate (0);
        i--;
        /* There is at least one "full" group of insns. */
        if (group_length == (full_group_size-1))
            full_group = true;
      }else if (id_right == id_left)
          grouped_insns->insns[j].ordered_remove (0);
    }
  }
  if (!full_group)
    return false;
  /* Remove groups which aren't full groups. 
    FORNOW: Groups are formed out of the instructions that work with the same
    data widths, and we're supporting merging only full groups. I.e. if four 
    instructions operate on two-byte data through the eight-byte registers, 
    they will be considered as a full group and could be merged into one 
    instruction. But, merging two instructions of that kind currently is not 
    supported. */
  for (size_t i = 0; i < grouped_insns->insns.length (); i++){
    size_t full_group_size = GET_MODE_SIZE (grouped_insns->vector_modes[i])/
                             GET_MODE_SIZE (grouped_insns->scalar_modes[i]);
    if (grouped_insns->insns[i].length () != full_group_size)
        grouped_insns->insns[i].truncate (0);
  }
  return true;
}

bool pass_merge_loadstores::adjacent_addresses (struct address_info *addr_info_a,
  struct address_info *addr_info_b, machine_mode *s_mode, machine_mode *v_mode) {
  /* Scalar mode for both insns. */
  *s_mode = addr_info_a->mode;
  int size = GET_MODE_SIZE (addr_info_a->mode);
  /* Investigate addr_info.index_term and addr_info.segment_term?? */
  if (addr_info_a->base_term && addr_info_b->base_term) {
    rtx rtx_a = *addr_info_a->base_term, rtx_b = *addr_info_b->base_term;
    if (REG_P (rtx_a) && REG_P (rtx_b) && REGNO (rtx_a) == REGNO (rtx_b))
      {
        /* Insns are using wider registers to load smaller data. 
           In case of merging a group of such insns, we'll use 
           register more - V_MODE for loading from memory.*/
        *v_mode = GET_MODE (rtx_a);
        HOST_WIDE_INT disp_a, disp_b;
        if (addr_info_a->disp_term)
          disp_a = XWINT (*addr_info_a->disp_term, 0);
        if (addr_info_b->disp_term)
          disp_b = XWINT (*addr_info_b->disp_term, 0);
        /* Check if insns're loading from adjacent adresses. */
        if (!addr_info_a->disp_term && disp_b == size)
          return true;
        if (!addr_info_b->disp_term && disp_a == size)
          return true;
        if (addr_info_a->disp_term && addr_info_b->disp_term && 
            ((disp_a - disp_b == size) || (disp_b-disp_a == size)))
          return true;
      }
  }
  return false;
}

/* Check if two insns access to adjacent memory. */
bool pass_merge_loadstores::adjacent (rtx_insn *i_a, rtx_insn *i_b, bool *load, 
  machine_mode *s_mode, machine_mode *v_mode) {
  rtx src_a, dst_a, src_b, dst_b;
  /* We have already checked that insn candidates are SET insns with the following forms:
   (set (mem ... )
        (subreg ... )
   ) - for stores,
   (set (reg ... )
        (zero_extend ... (mem ... ))
   ) - for loads. */
  src_a = SET_SRC (PATTERN (i_a));
  dst_a = SET_DEST (PATTERN (i_a));
  src_b = SET_SRC (PATTERN (i_b));
  dst_b = SET_DEST (PATTERN (i_b));
  bool store_a = MEM_P (dst_a);
  bool store_b = MEM_P (dst_b);

  struct address_info addr_info_a;
  struct address_info addr_info_b;

  if ((!store_a && store_b) || (store_a && !store_b))
    return false;

  /* Two load insns. */
  if (!store_a && !store_b){
    *load = true;
    decompose_mem_address (&addr_info_a, XEXP (src_a,0));
    decompose_mem_address (&addr_info_b, XEXP (src_b,0));
  }    
  /* Two store insns. */
  else {
    *load = false;
    decompose_mem_address (&addr_info_a, dst_a);
    decompose_mem_address (&addr_info_b, dst_b);
  }
  if (!addr_info_a.mode || !addr_info_b.mode
    || addr_info_a.mode != addr_info_b.mode)
    return false;
    
  return adjacent_addresses (&addr_info_a, &addr_info_b, s_mode, v_mode);
}

/* Instructions in the pair are named as left and right. One instruction can 
  belong to two pairs if it's declared as right in one pair, and left in 
  other pair. */
bool pass_merge_loadstores::already_paired (vec<std::pair<int,int>> *paired_insns, 
  int left_id, int right_id) {
  int i = 0;
  while (i < (*paired_insns).length ()) {
    if ((*paired_insns)[i].first == left_id 
      || (*paired_insns)[i].second == right_id)
      return true;
    i++;
  }
  return false;
}

/* Make pairs of two loads or two stores that can be merged. Those pairs are
   then added to the appropriate group - GROUPED_LOADS, or GROUPED_STORES. */
void pass_merge_loadstores::pair_insns (insn_vec &insn_candidates, vec<int> &insn_groups,
struct grouped_insns *grouped_loads, struct grouped_insns *grouped_stores) {
  auto_vec<std::pair<int, int>> paired_loads;
  auto_vec<std::pair<int, int>> paired_stores;

  /* Compare every two insns from the group and try to pair them. */
  for (size_t i = 0; i < insn_candidates.length () - 1; i++) {
    for (size_t j = i+1; j < insn_candidates.length (); j++) {
      bool load = false;
      machine_mode s_mode, v_mode;
      rtx_insn *i_a = insn_candidates[i];
      rtx_insn *i_b = insn_candidates[j];

      /* Two insns can be paired if both are load or store, if they belong to the 
         same group, if they access adjacent memory and aren't already paired. */
      if (insn_groups[i] != insn_groups[j] || 
          !adjacent (i_a, i_b, &load, &s_mode, &v_mode))
        continue;

      vec<rtx_insn *> new_pair = vNULL;
      new_pair.safe_push (i_a);
      new_pair.safe_push (i_b);

      if (!load) {
        if (!already_paired (&paired_stores, INSN_UID (i_a), INSN_UID (i_b))){
          paired_stores.safe_push ({INSN_UID (i_a), INSN_UID (i_b)});
          /* Each group of insns has matching SCALAR_MODES and VECTOR_MODES
           vectors.*/
          grouped_stores->insns.safe_push (new_pair);
          grouped_stores->scalar_modes.safe_push (s_mode);
          grouped_stores->vector_modes.safe_push (v_mode);
        }
      }else {
        if (!already_paired (&paired_loads, INSN_UID (i_a), INSN_UID (i_b))) {
          paired_loads.safe_push ({INSN_UID (i_a), INSN_UID (i_b)});
          grouped_loads->insns.safe_push (new_pair);
          grouped_loads->scalar_modes.safe_push (s_mode);
          grouped_loads->vector_modes.safe_push (v_mode);          
        }
      }
    }
  }
}

/* Try to merge sequences of insns with the same group id. */
bool pass_merge_loadstores::slp_analyze_bb (insn_vec &insn_candidates, vec<int> &insn_groups,
struct grouped_insns *grouped_loads, struct grouped_insns *grouped_stores) {
  /* First, make pairs of two loads or two stores accessing adjacent memory. */
  pair_insns (insn_candidates, insn_groups, grouped_loads, grouped_stores);
  /* If we don't have at least one store group, there's no need for further analysis. */
  if (grouped_stores->insns.length () == 0)
    return false;
  if (!merge_full_groups (grouped_stores))
    return false;
  merge_full_groups (grouped_loads);
  return true;
}

/* Instruction is a candidate for merging if it's a SET instruction containing 
   exactly one memory reference. For stores, we're interested in insns having
   the following form:
   (set (mem ... )
        (subreg ... )
   )
   For loads, we're interested in insns having the following form:
   (set (reg ... )
        (zero_extend ... (mem ... ))
   )
   The sequence of candidates is given as an INSN_CANDIDATES. We're tracking
   goups through the INSN_GROUPS. */
bool pass_merge_loadstores::is_candidate (rtx_insn *in, insn_vec &insn_candidates,
	vec<int> &insn_groups, int group_id) {
  if (!NONJUMP_INSN_P (in))
    return false;
  rtx pat = PATTERN (in);
  if (GET_CODE (pat) != SET)
    return false;
  rtx src, dst;
  src = SET_SRC (pat);
  dst = SET_DEST (pat);

  /* TODO: Check if this is necessary. */
  bool src_mem_ref = contains_mem_rtx_p (src), 
       dst_mem_ref = contains_mem_rtx_p (dst);
  if ((src_mem_ref && dst_mem_ref) || (!src_mem_ref && !dst_mem_ref))
    return false;

  /* Store insns working with smaller data than register size will have 
     SUBREG rtx for their source. */
  bool store = SUBREG_P (src) && MEM_P (dst);
  
  /* Load insns working with smaller data than register size will have 
     ZERO_EXTEND code for their source. */
  bool load = GET_CODE (src) == ZERO_EXTEND &&
             MEM_P (XEXP (src,0)) && REG_P (dst);

  if (!load && !store)
    return false;

  insn_candidates.safe_push (in);
  insn_groups.safe_push (group_id);
  return true;
}

void pass_merge_loadstores::assign_group_id_to_insn (basic_block bb, 
  insn_vec &insn_candidates, vec<int> &insn_groups) {
  rtx_insn *insn;
  int current_group = 0;

  FOR_BB_INSNS (bb, insn) {
    if (DEBUG_INSN_P (insn))
	    continue;
    if (!is_candidate (insn, insn_candidates, insn_groups, current_group)) {
      ++current_group;
    }
  }
}

/* Try to merge groups of loads and stores into one load/store insn. 
   Store groups can be merged independently, but each load group
   needs a matching store group. */
void pass_merge_loadstores::slp_bb (basic_block bb) {
  insn_vec insn_candidates;
  auto_vec<int> insn_groups;
  /* For each insn in the BB check if an insn is a candidate for slp. 
  Assign a group id to each candidate. The sequence of candidates will have a 
  unique group id. */
  assign_group_id_to_insn (bb, insn_candidates, insn_groups);
  /* If there are no candidates in this bb, just return. */
  if (insn_candidates.length () == 0)
    return;
  struct grouped_insns grouped_loads (true);
  struct grouped_insns grouped_stores (false);
  /* Result of analyzis is information if we manage to construct groups that we can merge. 
     Information for group transformation is given in GROUPED_LOADS and GROUPED_STORES.*/
  if (!slp_analyze_bb (insn_candidates, insn_groups, &grouped_loads, &grouped_stores))
    return;
  
  /* TODO: Count profitability. */

  insn_to_insn_info_map scheduled;
  auto_vec<std::pair<int,int>> matched_load_stores;
  bool first_insn = false; 
  for (rtl_ssa::insn_info *in : crtl->ssa->bb (bb)->nondebug_insns ()){
    if (!in->rtl () || in->is_artificial () || in->has_volatile_refs () || in->is_asm ())
      continue;
    rtx_insn *ins = in->rtl ();
    /* Schedule first insn. */
    if (NONJUMP_INSN_P (ins) && !first_insn) {
      first_insn = true;
      scheduled.put (ins, in);
      continue;
    }
    if (!try_to_schedule_ins (in, scheduled))
        continue;
    int grouped_load_group_i = grouped_ins (ins, &grouped_loads);
    int grouped_store_group_i = grouped_ins (ins, &grouped_stores);
    bool load_insn = grouped_load_group_i != -1, 
         store_insn = grouped_store_group_i != -1;
    if (!load_insn && !store_insn)
      continue;

    /* Grouped load insn. */
    if (load_insn) {
	    /* Check if every load insn in the group is scheduled. If so,
          we can try swapping all scalar load insns for the SIMD one.*/
      if (!group_all_scheduled (scheduled, &grouped_loads, grouped_load_group_i))
          continue;
      
      /* Find a matching store group. */
      int full_group_size = GET_MODE_SIZE (grouped_loads.vector_modes[grouped_load_group_i])/
                            GET_MODE_SIZE (grouped_loads.scalar_modes[grouped_load_group_i]);
      int matching_store_idx = matching_store_group (scheduled, 
        grouped_loads.insns[grouped_load_group_i], &grouped_stores, full_group_size);
      /* If there's no matching store group, we can't perform merging. */
      if (matching_store_idx == -1)
        continue;
      matched_load_stores.safe_push ({grouped_load_group_i, matching_store_idx});
      
      /* Transform load group. */
      rtx_insn *first_load_in_group = grouped_loads.insns[grouped_load_group_i][0];
      rtl_ssa::insn_info *insn_to_change = *(scheduled.get (first_load_in_group));
      transform (&grouped_loads, grouped_load_group_i, insn_to_change, bb);
    }

    /* Grouped store insn. */
    else {
	    /* Check if every ins in group is scheduled. If so, we can swap scalar insns 
      for SIMD one.*/       
      if (!group_all_scheduled (scheduled, &grouped_stores, grouped_store_group_i))
        continue;
      
      rtx_insn *first_store_in_group = grouped_stores.insns[grouped_store_group_i][0];             
      bool store_group_swapped = false;
      /* First, try to find a matching load group. */
      for (size_t i = 0; i < matched_load_stores.length (); i++)
        if (matched_load_stores[i].second == grouped_store_group_i) { 
          rtl_ssa::insn_info *insn_to_change = *(scheduled.get (first_store_in_group));
          transform (&grouped_stores, grouped_store_group_i, insn_to_change, bb);
          store_group_swapped = true;
          break;
        }

      if (store_group_swapped)
        continue;

      /* If there's no matching load group, we can still transform store group. 
        We'll need extra instructions to load all data from multiple registers into one. */
      /* TODO: Cover the case when we can SLP group of stores, without matching load group. */
    }
  }
}

unsigned int
pass_merge_loadstores::execute (function *fn) {
  calculate_dominance_info (CDI_DOMINATORS);
  crtl->ssa = new rtl_ssa::function_info (cfun);
  basic_block bb;
  FOR_ALL_BB_FN (bb, fn) {
    slp_bb (bb);
  }
  free_dominance_info (CDI_DOMINATORS);
  if (crtl->ssa->perform_pending_updates ())
    cleanup_cfg (0);
  delete crtl->ssa;
  crtl->ssa = nullptr;
  return 0;
}

} // anon namespace

rtl_opt_pass *
make_pass_merge_loadstores (gcc::context *ctxt)
{
  return new pass_merge_loadstores (ctxt);
}
